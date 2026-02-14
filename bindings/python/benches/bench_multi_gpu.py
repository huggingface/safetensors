"""Multi-GPU model loading benchmark.

Compares:
  1. ours cuFile (device_map)    — safe_open(index.json, device=map, backend="cufile")
  2. ours io_uring (device_map)  — safe_open(index.json, device=map, backend="io_uring")
  3. transformers                — from_pretrained(device_map=map)
  4. fastsafetensors             — 2 loaders with Python threads, one per GPU

Usage:
  CUDA_VISIBLE_DEVICES=0,2 LD_LIBRARY_PATH=... python bench_multi_gpu.py
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import threading
from pathlib import Path

import torch
from huggingface_hub import snapshot_download


def parse_args():
    p = argparse.ArgumentParser(description="Multi-GPU model loading benchmark")
    p.add_argument("--model", default="Qwen/Qwen2.5-14B", help="HuggingFace model ID")
    p.add_argument("--cache-dir", default="/raid/hf_cache", help="HF cache directory")
    p.add_argument("--runs", type=int, default=5, help="Benchmark iterations")
    p.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    p.add_argument("--skip-fastsafetensors", action="store_true")
    p.add_argument("--skip-transformers", action="store_true")
    p.add_argument("--transformers-venv", default=os.path.expanduser("~/.venv/transformers"),
                   help="Path to transformers venv (uses upstream safetensors, not our fork)")
    return p.parse_args()


def resolve_model(model_id, cache_dir):
    model_path = snapshot_download(
        model_id, cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json"],
        ignore_patterns=["*.bin", "*.pt", "*.onnx"],
    )
    index_file = Path(model_path) / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            idx = json.load(f)
        sf_files = sorted(set(idx["weight_map"].values()))
        sf_files = [Path(model_path) / f for f in sf_files]
    else:
        sf_files = sorted(Path(model_path).glob("*.safetensors"))
    return model_path, sf_files


def build_tensor_device_map(index_file):
    """Build a simple round-robin tensor→device map for 2 GPUs."""
    with open(index_file) as f:
        idx = json.load(f)
    tensor_names = sorted(idx["weight_map"].keys())
    return {name: f"cuda:{i % 2}" for i, name in enumerate(tensor_names)}


def get_model_size_gb(sf_files):
    return sum(f.stat().st_size for f in sf_files) / (1024**3)


def sync_gpus():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def clear_gpu_memory():
    gc.collect()
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()


# ── Benchmark: ours (safe_open with device_map) ───────────────────────────────

def bench_ours(index_file, device_map, backend, n_warmup, n_runs):
    from safetensors import safe_open

    tensor_names = sorted(device_map.keys())
    times = []

    for i in range(n_warmup + n_runs):
        clear_gpu_memory()
        sync_gpus()
        t0 = time.perf_counter()

        with safe_open(str(index_file), framework="pt", device=device_map, backend=backend) as sf:
            for name in tensor_names:
                _ = sf.get_tensor(name)

        sync_gpus()
        elapsed = time.perf_counter() - t0

        if i >= n_warmup:
            times.append(elapsed)

    return times


# ── Benchmark: transformers (subprocess with separate venv) ───────────────────

def bench_transformers(model_path, device_map, n_warmup, n_runs, venv_path):
    """Benchmark transformers from_pretrained with device_map.

    Runs in a subprocess using the transformers venv (upstream safetensors).
    """
    # Convert tensor-level map to module-level map for transformers
    module_map = {}
    for tensor_name, device in device_map.items():
        if tensor_name.endswith(".weight") or tensor_name.endswith(".bias"):
            module_name = tensor_name.rsplit(".", 1)[0]
        else:
            module_name = tensor_name
        module_map[module_name] = device

    script = f'''
import gc, json, time, torch

model_path = {model_path!r}
module_map = json.loads({json.dumps(json.dumps(module_map))})
n_warmup = {n_warmup}
n_runs = {n_runs}

# Init CUDA
for i in range(torch.cuda.device_count()):
    _ = torch.zeros(1, device=f"cuda:{{i}}")
    torch.cuda.synchronize(i)

from transformers import AutoModelForCausalLM

times = []
for i in range(n_warmup + n_runs):
    gc.collect()
    for j in range(torch.cuda.device_count()):
        with torch.cuda.device(j):
            torch.cuda.empty_cache()
        torch.cuda.synchronize(j)

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map=module_map, torch_dtype=torch.bfloat16,
    )
    for j in range(torch.cuda.device_count()):
        torch.cuda.synchronize(j)
    elapsed = time.perf_counter() - t0

    if i >= n_warmup:
        times.append(elapsed)
    del model
    gc.collect()
    for j in range(torch.cuda.device_count()):
        with torch.cuda.device(j):
            torch.cuda.empty_cache()

# Output as JSON
print("BENCH_RESULT:" + json.dumps(times))
'''

    python_bin = os.path.join(venv_path, "bin", "python")
    env = os.environ.copy()
    result = subprocess.run(
        [python_bin, "-c", script],
        capture_output=True, text=True, env=env, timeout=600,
    )

    if result.returncode != 0:
        raise RuntimeError(f"transformers subprocess failed:\n{result.stderr}")

    for line in result.stdout.splitlines():
        if line.startswith("BENCH_RESULT:"):
            return json.loads(line[len("BENCH_RESULT:"):])

    raise RuntimeError(f"No BENCH_RESULT in output:\n{result.stdout}\n{result.stderr}")


# ── Benchmark: fastsafetensors ────────────────────────────────────────────────

def bench_fastsafetensors(sf_files, device_map, n_warmup, n_runs, use_gds=True):
    """Benchmark fastsafetensors with 2 loaders (one per GPU), threaded.

    fastsafetensors doesn't support per-tensor device maps in a single process.
    We create one loader per GPU and load all files to both, then extract the
    relevant tensors from each. This is the closest single-process equivalent.
    """
    from fastsafetensors import SafeTensorsFileLoader

    tensor_names = sorted(device_map.keys())
    # Group tensors by target device
    tensors_per_device = {}
    for name, dev in device_map.items():
        tensors_per_device.setdefault(dev, []).append(name)

    file_list = [str(f) for f in sf_files]
    times = []

    for i in range(n_warmup + n_runs):
        clear_gpu_memory()
        sync_gpus()
        t0 = time.perf_counter()

        # Create one loader per device
        loaders = {}
        buffers = {}
        for dev_str in sorted(tensors_per_device.keys()):
            loader = SafeTensorsFileLoader(
                pg=None, device=dev_str,
                nogds=not use_gds, max_threads=16,
            )
            loader.add_filenames({0: file_list})
            loaders[dev_str] = loader

        # Load files to both devices concurrently (GIL released in C++)
        def load_device(dev_str):
            buffers[dev_str] = loaders[dev_str].copy_files_to_device()

        threads = []
        for dev_str in loaders:
            t = threading.Thread(target=load_device, args=(dev_str,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        # Extract tensors from the correct device's buffer
        tensors = {}
        for dev_str, names in tensors_per_device.items():
            buf = buffers[dev_str]
            for name in names:
                tensors[name] = buf.get_tensor(name)

        sync_gpus()
        elapsed = time.perf_counter() - t0

        if i >= n_warmup:
            times.append(elapsed)

        # Cleanup
        del tensors
        for buf in buffers.values():
            buf.close()
        for loader in loaders.values():
            loader.close()
        del buffers, loaders
        clear_gpu_memory()

    return times


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    assert torch.cuda.device_count() >= 2, \
        f"Need >= 2 GPUs, got {torch.cuda.device_count()}. Set CUDA_VISIBLE_DEVICES=X,Y"

    print(f"Resolving model: {args.model}")
    model_path, sf_files = resolve_model(args.model, args.cache_dir)
    model_size = get_model_size_gb(sf_files)

    index_file = Path(model_path) / "model.safetensors.index.json"
    assert index_file.exists(), f"No index file: {index_file}"

    device_map = build_tensor_device_map(index_file)
    n_tensors = len(device_map)
    per_dev = {}
    for d in device_map.values():
        per_dev[d] = per_dev.get(d, 0) + 1

    # Initialize CUDA
    for i in range(torch.cuda.device_count()):
        _ = torch.zeros(1, device=f"cuda:{i}")
    sync_gpus()

    gpu_names = [torch.cuda.get_device_name(i) for i in range(2)]

    print("=" * 80)
    print("Multi-GPU Model Loading Benchmark")
    print("=" * 80)
    print(f"  Model:   {args.model} ({model_size:.1f} GB, {len(sf_files)} shards)")
    print(f"  GPUs:    cuda:0 ({gpu_names[0]}), cuda:1 ({gpu_names[1]})")
    print(f"  Tensors: {n_tensors} total — {per_dev}")
    print(f"  Warmup:  {args.warmup}, Runs: {args.runs}")
    print("=" * 80)

    results = {}

    # ── Ours: cuFile ──
    print(f"\n--- ours cuFile (device_map) ---")
    try:
        times = bench_ours(index_file, device_map, "cufile", args.warmup, args.runs)
        avg = sum(times) / len(times) * 1000
        best = min(times) * 1000
        ms_list = [int(t * 1000) for t in times]
        print(f"  avg {avg:7.0f}ms  best {best:7.0f}ms  {model_size / (best/1000):5.1f} GB/s  {ms_list}")
        results["ours_cufile"] = (avg, best)
    except Exception as e:
        print(f"  FAILED: {e}")

    # ── Ours: io_uring ──
    print(f"\n--- ours io_uring (device_map) ---")
    try:
        times = bench_ours(index_file, device_map, "io_uring", args.warmup, args.runs)
        avg = sum(times) / len(times) * 1000
        best = min(times) * 1000
        ms_list = [int(t * 1000) for t in times]
        print(f"  avg {avg:7.0f}ms  best {best:7.0f}ms  {model_size / (best/1000):5.1f} GB/s  {ms_list}")
        results["ours_iouring"] = (avg, best)
    except Exception as e:
        print(f"  FAILED: {e}")

    # ── fastsafetensors (GDS) ──
    if not args.skip_fastsafetensors:
        print(f"\n--- fastsafetensors GDS (2 loaders, threaded) ---")
        try:
            times = bench_fastsafetensors(sf_files, device_map, args.warmup, args.runs, use_gds=True)
            avg = sum(times) / len(times) * 1000
            best = min(times) * 1000
            ms_list = [int(t * 1000) for t in times]
            print(f"  avg {avg:7.0f}ms  best {best:7.0f}ms  {model_size / (best/1000):5.1f} GB/s  {ms_list}")
            results["fst_gds"] = (avg, best)
        except Exception as e:
            print(f"  FAILED: {e}")

        print(f"\n--- fastsafetensors no-GDS (2 loaders, threaded) ---")
        try:
            times = bench_fastsafetensors(sf_files, device_map, args.warmup, args.runs, use_gds=False)
            avg = sum(times) / len(times) * 1000
            best = min(times) * 1000
            ms_list = [int(t * 1000) for t in times]
            print(f"  avg {avg:7.0f}ms  best {best:7.0f}ms  {model_size / (best/1000):5.1f} GB/s  {ms_list}")
            results["fst_nogds"] = (avg, best)
        except Exception as e:
            print(f"  FAILED: {e}")

    # ── transformers (separate venv with upstream safetensors) ──
    if not args.skip_transformers:
        print(f"\n--- transformers from_pretrained(device_map=...) ---")
        print(f"    (using {args.transformers_venv} with upstream safetensors)")
        try:
            times = bench_transformers(model_path, device_map, args.warmup, args.runs, args.transformers_venv)
            avg = sum(times) / len(times) * 1000
            best = min(times) * 1000
            ms_list = [int(t * 1000) for t in times]
            print(f"  avg {avg:7.0f}ms  best {best:7.0f}ms  {model_size / (best/1000):5.1f} GB/s  {ms_list}")
            results["transformers"] = (avg, best)
        except Exception as e:
            print(f"  FAILED: {e}")

    # ── Summary ──
    print(f"\n{'='*80}")
    print("Summary (sorted by avg time)")
    print(f"{'='*80}")
    sorted_results = sorted(results.items(), key=lambda x: x[1][0])
    if sorted_results:
        fastest_avg = sorted_results[0][1][0]
        for name, (avg, best) in sorted_results:
            ratio = avg / fastest_avg
            tag = " <-- fastest" if ratio == 1.0 else ""
            print(f"  {name:<30} {avg:7.0f}ms   {model_size / (best/1000):5.1f} GB/s  ({ratio:.2f}x){tag}")

    print("\nDone!")


if __name__ == "__main__":
    main()

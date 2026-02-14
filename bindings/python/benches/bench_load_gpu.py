"""
Benchmark: GPU model loading — safetensors vs fastsafetensors vs transformers.

Loads a sharded safetensors model to a single GPU and compares throughput
across all available backends and loading strategies.

All benchmarks force data materialization with .sum() on every tensor
followed by torch.cuda.synchronize() to ensure data is fully on GPU
before stopping the timer.

Methods compared:
  1. fastsafetensors pread    — pread() + cudaMemcpy, 16 internal threads
  2. fastsafetensors cuFile   — cuFileRead direct NVMe→GPU
  3. ours io_uring (sharded)  — safe_open(index.json), Rust spawns N threads
  4. ours cuFile (sharded)    — same but cuFileRead per thread
  5. ours io_uring (sequential) — Python for-loop over shard files
  6. ours cuFile (sequential)   — same but cuFile backend
  7. transformers AutoModel    — from_pretrained() (separate venv)

Requirements:
  - torch, safetensors, fastsafetensors, huggingface_hub
  - fastsafetensors needs torch.distributed (auto-initialized with world_size=1)
  - transformers benchmark runs in a separate venv at ~/.venv/transformers

Usage:
  # Default: Qwen2.5-14B, cache in /raid/hf_cache
  LD_LIBRARY_PATH=$HOME/.venv/safetensors/lib/python3.12/site-packages/nvidia/cuda_runtime/lib \
    CUDA_VISIBLE_DEVICES=0 \
    ~/.venv/safetensors/bin/python benches/bench_load_gpu.py

  # Different model:
  LD_LIBRARY_PATH=$HOME/.venv/safetensors/lib/python3.12/site-packages/nvidia/cuda_runtime/lib \
    CUDA_VISIBLE_DEVICES=0 \
    ~/.venv/safetensors/bin/python benches/bench_load_gpu.py --model openai/gpt-oss-20b

  # Skip transformers (slow):
  LD_LIBRARY_PATH=$HOME/.venv/safetensors/lib/python3.12/site-packages/nvidia/cuda_runtime/lib \
    CUDA_VISIBLE_DEVICES=0 \
    ~/.venv/safetensors/bin/python benches/bench_load_gpu.py --skip-transformers
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


def parse_args():
    p = argparse.ArgumentParser(description="GPU model loading benchmark")
    p.add_argument("--model", default="Qwen/Qwen2.5-14B", help="HuggingFace model ID")
    p.add_argument("--cache-dir", default="/raid/hf_cache", help="HF cache directory")
    p.add_argument("--device", default="cuda:0", help="Target GPU device")
    p.add_argument("--runs", type=int, default=5, help="Benchmark iterations")
    p.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    p.add_argument("--skip-fastsafetensors", action="store_true")
    p.add_argument("--skip-transformers", action="store_true")
    p.add_argument("--transformers-venv", default=os.path.expanduser("~/.venv/transformers"))
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


def materialize(d):
    for t in d.values():
        float(t.sum())


def fmt(label, times, total_bytes):
    avg = sum(times) / len(times)
    best = min(times)
    tp = total_bytes / avg / 1024**3
    runs_str = ", ".join(f"{t * 1000:.0f}" for t in times)
    print(f"  {label:40s}  avg {avg * 1000:7.0f}ms  best {best * 1000:7.0f}ms  {tp:5.1f} GB/s  [{runs_str}]")
    return avg


# ---------------------------------------------------------------------------
# fastsafetensors
# ---------------------------------------------------------------------------
def bench_fastsafetensors(label, sf_files, device, total_bytes, nogds, warmup, runs):
    import torch.distributed as dist
    from fastsafetensors import SafeTensorsFileLoader

    loader = SafeTensorsFileLoader(
        pg=dist.group.WORLD, device=device, nogds=nogds, debug_log=False)

    for _ in range(warmup):
        loader.add_filenames({0: [str(f) for f in sf_files]})
        buf = loader.copy_files_to_device()
        result = {k: buf.get_tensor(k) for k in loader.get_keys()}
        materialize(result)
        torch.cuda.synchronize()
        buf.close(); loader.reset()
        del result; torch.cuda.empty_cache(); gc.collect()

    times = []
    for _ in range(runs):
        torch.cuda.synchronize(); torch.cuda.empty_cache(); gc.collect()
        t0 = time.perf_counter()
        loader.add_filenames({0: [str(f) for f in sf_files]})
        buf = loader.copy_files_to_device()
        result = {k: buf.get_tensor(k) for k in loader.get_keys()}
        materialize(result)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        buf.close(); loader.reset(); del result

    loader.close()
    return fmt(label, times, total_bytes)


# ---------------------------------------------------------------------------
# safetensors sharded (single safe_open on index.json — Rust threads)
# ---------------------------------------------------------------------------
def bench_sharded(label, index_path, device, total_bytes, backend, warmup, runs):
    for _ in range(warmup):
        with safe_open(index_path, framework="pt", device=device, backend=backend) as sf:
            result = {name: sf.get_tensor(name) for name in sf.keys()}
        materialize(result)
        torch.cuda.synchronize()
        del result; torch.cuda.empty_cache(); gc.collect()

    times = []
    for _ in range(runs):
        torch.cuda.synchronize(); torch.cuda.empty_cache(); gc.collect()
        t0 = time.perf_counter()
        with safe_open(index_path, framework="pt", device=device, backend=backend) as sf:
            result = {name: sf.get_tensor(name) for name in sf.keys()}
        materialize(result)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        del result

    return fmt(label, times, total_bytes)


# ---------------------------------------------------------------------------
# safetensors sequential (Python for-loop over shard files)
# ---------------------------------------------------------------------------
def bench_sequential(label, sf_files, device, total_bytes, backend, warmup, runs):
    for _ in range(warmup):
        result = {}
        for sf_file in sf_files:
            with safe_open(str(sf_file), framework="pt", device=device, backend=backend) as sf:
                for name in sf.keys():
                    result[name] = sf.get_tensor(name)
        materialize(result)
        torch.cuda.synchronize()
        del result; torch.cuda.empty_cache(); gc.collect()

    times = []
    for _ in range(runs):
        torch.cuda.synchronize(); torch.cuda.empty_cache(); gc.collect()
        t0 = time.perf_counter()
        result = {}
        for sf_file in sf_files:
            with safe_open(str(sf_file), framework="pt", device=device, backend=backend) as sf:
                for name in sf.keys():
                    result[name] = sf.get_tensor(name)
        materialize(result)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        del result

    return fmt(label, times, total_bytes)


# ---------------------------------------------------------------------------
# transformers AutoModel (runs in separate venv)
# ---------------------------------------------------------------------------
def bench_transformers(model_id, cache_dir, device, total_bytes, venv, warmup, runs):
    python = os.path.join(venv, "bin", "python")
    if not os.path.exists(python):
        print(f"  [SKIP] transformers venv not found at {venv}")
        return None

    script = f"""
import gc, time, json, os, torch
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
LD = "{venv}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
if LD not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = LD + ":" + os.environ.get("LD_LIBRARY_PATH", "")
from transformers import AutoModelForCausalLM
for _ in range({warmup}):
    model = AutoModelForCausalLM.from_pretrained(
        "{model_id}", device_map={{\"\":\"{device}\"}}, dtype=torch.float16,
        cache_dir={json.dumps(cache_dir)})
    torch.cuda.synchronize()
    del model; torch.cuda.empty_cache(); gc.collect()
times = []
for _ in range({runs}):
    torch.cuda.synchronize(); torch.cuda.empty_cache(); gc.collect()
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        "{model_id}", device_map={{\"\":\"{device}\"}}, dtype=torch.float16,
        cache_dir={json.dumps(cache_dir)})
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)
    del model
print(json.dumps(times))
"""
    env = os.environ.copy()
    cuda_lib = f"{venv}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
    env["LD_LIBRARY_PATH"] = cuda_lib + ":" + env.get("LD_LIBRARY_PATH", "")

    result = subprocess.run(
        [python, "-c", script],
        capture_output=True, text=True, env=env, timeout=600,
    )
    if result.returncode != 0:
        print(f"  [ERROR] {result.stderr[-500:]}")
        return None

    for line in reversed(result.stdout.strip().split("\n")):
        try:
            times = json.loads(line)
            return fmt("transformers from_pretrained", times, total_bytes)
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def main():
    args = parse_args()

    model_path, sf_files = resolve_model(args.model, args.cache_dir)
    index_path = str(Path(model_path) / "model.safetensors.index.json")
    has_index = os.path.exists(index_path)

    total_bytes = sum(os.path.getsize(f) for f in sf_files)
    total_gb = total_bytes / 1024**3

    print("=" * 90)
    print(f"GPU Model Loading Benchmark")
    print("=" * 90)
    print(f"  Model:   {args.model} ({total_gb:.1f} GB, {len(sf_files)} shards)")
    print(f"  Device:  {args.device} ({torch.cuda.get_device_name()})")
    print(f"  Warmup:  {args.warmup}, Runs: {args.runs}")
    print(f"  Index:   {'yes' if has_index else 'no'}")
    print("=" * 90)

    results = {}

    # --- fastsafetensors ---
    if not args.skip_fastsafetensors:
        # Initialize torch.distributed (fastsafetensors requires it)
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", rank=0, world_size=1)

        print()
        print("--- fastsafetensors ---")
        print("    All shards submitted at once, internal threading")
        try:
            results["fst_pread"] = bench_fastsafetensors(
                "fst pread (nogds=True)", sf_files, args.device,
                total_bytes, nogds=True, warmup=args.warmup, runs=args.runs)
        except Exception as e:
            print(f"  [ERROR] {e}")

        try:
            results["fst_cufile"] = bench_fastsafetensors(
                "fst cuFile (nogds=False)", sf_files, args.device,
                total_bytes, nogds=False, warmup=args.warmup, runs=args.runs)
        except Exception as e:
            print(f"  [ERROR] {e}")

    # --- safetensors sharded ---
    if has_index:
        print()
        print("--- safetensors sharded (1 safe_open call, Rust threads) ---")
        print(f"    safe_open(index.json) → Rust spawns {len(sf_files)} threads")
        try:
            results["ours_iouring_sharded"] = bench_sharded(
                "ours io_uring (sharded)", index_path, args.device,
                total_bytes, "io_uring", args.warmup, args.runs)
        except Exception as e:
            print(f"  [ERROR] {e}")

        try:
            results["ours_cufile_sharded"] = bench_sharded(
                "ours cuFile (sharded)", index_path, args.device,
                total_bytes, "cufile", args.warmup, args.runs)
        except Exception as e:
            print(f"  [ERROR] {e}")

    # --- safetensors sequential ---
    print()
    print("--- safetensors sequential (Python for-loop over files) ---")
    try:
        results["ours_iouring_seq"] = bench_sequential(
            "ours io_uring (sequential)", sf_files, args.device,
            total_bytes, "io_uring", args.warmup, args.runs)
    except Exception as e:
        print(f"  [ERROR] {e}")

    try:
        results["ours_cufile_seq"] = bench_sequential(
            "ours cuFile (sequential)", sf_files, args.device,
            total_bytes, "cufile", args.warmup, args.runs)
    except Exception as e:
        print(f"  [ERROR] {e}")

    # --- transformers ---
    if not args.skip_transformers:
        print()
        print("--- transformers ---")
        try:
            avg = bench_transformers(
                args.model, args.cache_dir, args.device, total_bytes,
                args.transformers_venv, args.warmup, args.runs)
            if avg is not None:
                results["transformers"] = avg
        except Exception as e:
            print(f"  [ERROR] {e}")

    # --- Summary ---
    print()
    print("=" * 90)
    print("Summary (sorted by throughput)")
    print("=" * 90)
    if results:
        fastest = min(results.values())
        for name, avg_s in sorted(results.items(), key=lambda x: x[1]):
            tp = total_gb / avg_s
            ratio = avg_s / fastest
            marker = " <-- fastest" if avg_s == fastest else ""
            print(f"  {name:30s}  {avg_s * 1000:7.0f}ms  {tp:5.1f} GB/s  ({ratio:.2f}x){marker}")

    # Cleanup
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    print()
    print("Done!")


if __name__ == "__main__":
    main()

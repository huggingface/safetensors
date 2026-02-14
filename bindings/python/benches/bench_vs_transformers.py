#!/usr/bin/env python3
"""Weight loading benchmark: safetensors io_uring scatter vs transformers.

Compares raw tensor loading time (first I/O to all tensors on target GPUs):
  1. transformers async — safe_open(cpu) + ThreadPoolExecutor(4) materialize+to(device)
  2. transformers sync  — safe_open(cpu) + sequential materialize+to(device)
  3. ours io_uring      — safe_open(index.json, device=map, backend="io_uring")
  4. ours mmap          — safe_open(index.json, device=map, backend="mmap")

Methods 1 & 2 run in a subprocess using ~/.venv/transformers (upstream safetensors)
to avoid import conflicts with our fork.

Usage:
  CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu \\
    ~/.venv/safetensors/bin/python bench_vs_transformers.py --cold-cache --runs 3
"""

import argparse
import gc
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download


DTYPE_SIZES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Weight loading benchmark: io_uring scatter vs transformers"
    )
    p.add_argument("--model", default="Qwen/Qwen1.5-110B-Chat",
                   help="HuggingFace model ID")
    p.add_argument("--cache-dir", default="/raid/hf_cache",
                   help="HF cache directory")
    p.add_argument("--n-gpus", type=int, default=None,
                   help="Number of GPUs to use (default: all visible)")
    p.add_argument("--runs", type=int, default=3,
                   help="Timed iterations per method")
    p.add_argument("--warmup", type=int, default=1,
                   help="Warmup iterations (not timed)")
    p.add_argument("--cold-cache", action="store_true",
                   help="Drop page cache before each run (requires sudo)")
    p.add_argument("--skip-transformers", action="store_true",
                   help="Skip transformers baseline methods")
    p.add_argument("--transformers-venv",
                   default=os.path.expanduser("~/.venv/transformers"),
                   help="Venv with upstream safetensors + transformers")
    return p.parse_args()


# ── Model resolution ─────────────────────────────────────────────────────────

def resolve_model(model_id, cache_dir):
    """Download/resolve model. Returns (model_path, shard_files, index_data)."""
    model_path = snapshot_download(
        model_id, cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json"],
        ignore_patterns=["*.bin", "*.pt", "*.onnx"],
    )
    index_file = Path(model_path) / "model.safetensors.index.json"
    assert index_file.exists(), f"No index file found at {index_file}"
    with open(index_file) as f:
        index_data = json.load(f)
    shard_names = sorted(set(index_data["weight_map"].values()))
    shard_files = [Path(model_path) / s for s in shard_names]
    return model_path, shard_files, index_data


def read_safetensors_metadata(path):
    """Parse safetensors file header -> {name: (dtype, shape)}."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    return {
        name: (info["dtype"], info["shape"])
        for name, info in header.items()
        if name != "__metadata__"
    }


def tensor_byte_size(dtype, shape):
    n = DTYPE_SIZES[dtype]
    for s in shape:
        n *= s
    return n


def build_balanced_device_map(model_path, index_data, n_gpus):
    """Size-balanced tensor->device assignment (greedy, largest-first).

    Returns (device_map, total_bytes, gpu_bytes, bytes_per_device).
    """
    weight_map = index_data["weight_map"]

    # Read metadata from each shard file
    shard_meta = {}
    for shard_file in set(weight_map.values()):
        shard_meta[shard_file] = read_safetensors_metadata(
            Path(model_path) / shard_file
        )

    # Compute tensor sizes
    tensor_sizes = {}
    for name, shard_file in weight_map.items():
        dtype, shape = shard_meta[shard_file][name]
        tensor_sizes[name] = tensor_byte_size(dtype, shape)

    # Greedy assignment: largest tensors first -> least-loaded GPU
    gpu_bytes = [0] * n_gpus
    device_map = {}
    for name in sorted(tensor_sizes, key=tensor_sizes.get, reverse=True):
        target = min(range(n_gpus), key=lambda i: gpu_bytes[i])
        device_map[name] = f"cuda:{target}"
        gpu_bytes[target] += tensor_sizes[name]

    total_bytes = sum(tensor_sizes.values())

    # Per-device byte totals (for caching_allocator_warmup)
    bytes_per_device = {}
    for name, dev in device_map.items():
        bytes_per_device[dev] = bytes_per_device.get(dev, 0) + tensor_sizes[name]

    return device_map, total_bytes, gpu_bytes, bytes_per_device


# ── Utilities ────────────────────────────────────────────────────────────────

def drop_page_cache():
    """Drop OS page cache. Requires passwordless sudo."""
    subprocess.run(
        ["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
        check=True, capture_output=True,
    )


def sync_gpus():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def clear_gpu_memory():
    gc.collect()
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()


# ── Benchmark: ours (safe_open with device_map) ─────────────────────────────

def bench_ours(index_file, device_map, backend, n_warmup, n_runs, cold_cache):
    from safetensors import safe_open

    times = []
    for i in range(n_warmup + n_runs):
        clear_gpu_memory()
        if cold_cache:
            drop_page_cache()
        sync_gpus()
        t0 = time.perf_counter()

        with safe_open(str(index_file), framework="pt",
                       device=device_map, backend=backend) as sf:
            for name in sf.keys():
                _ = sf.get_tensor(name)

        sync_gpus()
        elapsed = time.perf_counter() - t0

        label = "warmup" if i < n_warmup else f"run {i - n_warmup + 1}"
        print(f"    {label}: {elapsed*1000:.0f}ms", flush=True)
        if i >= n_warmup:
            times.append(elapsed)

    return times


# ── Benchmark: transformers pattern (subprocess) ────────────────────────────

def bench_transformers(shard_files, device_map, bytes_per_device,
                       mode, n_warmup, n_runs, cold_cache, venv_path):
    """Run transformers-style loading in a subprocess with upstream safetensors.

    mode: "async" (ThreadPoolExecutor(4)) or "sync" (sequential).
    """
    shard_list = [str(f) for f in shard_files]

    # Build the subprocess script as an f-string with double-json-dumps
    # for safe serialization of dicts/lists into the Python source
    script = f'''
import gc, json, os, subprocess, sys, time
import torch
from concurrent.futures import ThreadPoolExecutor
from safetensors import safe_open

shard_files = json.loads({json.dumps(json.dumps(shard_list))})
device_map = json.loads({json.dumps(json.dumps(device_map))})
bytes_per_device = json.loads({json.dumps(json.dumps(bytes_per_device))})
mode = {mode!r}
n_warmup = {n_warmup}
n_runs = {n_runs}
cold_cache = {cold_cache!r}

# Init CUDA contexts
for i in range(torch.cuda.device_count()):
    _ = torch.zeros(1, device=f"cuda:{{i}}")
    torch.cuda.synchronize(i)

tensor_names = sorted(device_map.keys())
times = []

for run_idx in range(n_warmup + n_runs):
    gc.collect()
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()

    if cold_cache:
        subprocess.run(["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
                       check=True, capture_output=True)

    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)
    t0 = time.perf_counter()

    # Phase 1: open each shard on CPU, collect slices (lazy, no data read yet)
    merged = {{}}
    handles = []
    for shard_file in shard_files:
        fp = safe_open(shard_file, framework="pt", device="cpu")
        handles.append(fp)
        for k in fp.keys():
            if k in device_map:
                merged[k] = fp.get_slice(k)

    # Phase 2: caching_allocator_warmup -- pre-allocate GPU memory so the
    # CUDA caching allocator doesn't incur kernel-level allocs during loading
    for dev_str, nbytes in bytes_per_device.items():
        if nbytes > 0:
            _ = torch.empty(nbytes // 2, dtype=torch.float16, device=dev_str)
            del _

    # Phase 3: materialize slices on CPU + transfer to target GPU
    if mode == "async":
        def _load(name):
            return merged[name][...].to(device_map[name])
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {{name: pool.submit(_load, name) for name in tensor_names}}
            tensors = {{name: f.result() for name, f in futures.items()}}
        del futures  # Futures hold result refs -> GPU tensors; must free before gc
    else:
        tensors = {{}}
        for name in tensor_names:
            tensors[name] = merged[name][...].to(device_map[name])

    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)
    elapsed = time.perf_counter() - t0

    label = "warmup" if run_idx < n_warmup else f"run {{run_idx - n_warmup + 1}}"
    print(f"    {{label}}: {{elapsed*1000:.0f}}ms", flush=True)
    if run_idx >= n_warmup:
        times.append(elapsed)

    del tensors, merged, handles
    gc.collect()
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()

print("BENCH_RESULT:" + json.dumps(times))
'''

    python_bin = os.path.join(venv_path, "bin", "python")
    env = os.environ.copy()
    result = subprocess.run(
        [python_bin, "-c", script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env, timeout=1800,
    )

    # Relay subprocess output (per-run timing lines)
    for line in result.stdout.splitlines():
        if not line.startswith("BENCH_RESULT:"):
            print(line, flush=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"subprocess failed (rc={result.returncode}):\n"
            f"{result.stderr[-3000:]}"
        )

    for line in result.stdout.splitlines():
        if line.startswith("BENCH_RESULT:"):
            return json.loads(line[len("BENCH_RESULT:"):])

    raise RuntimeError(
        f"No BENCH_RESULT in output:\n{result.stdout[-2000:]}"
    )


# ── Output formatting ───────────────────────────────────────────────────────

def fmt_result(times, total_bytes):
    """Format timing results. Returns (avg_s, best_s, formatted_line)."""
    avg_s = sum(times) / len(times)
    best_s = min(times)
    worst_s = max(times)
    gb = total_bytes / 1e9  # SI GB
    line = (
        f"  avg {avg_s*1000:7.0f}ms  best {best_s*1000:7.0f}ms  "
        f"worst {worst_s*1000:7.0f}ms  {gb/best_s:5.1f} GB/s"
    )
    return avg_s, best_s, line


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    n_gpus = args.n_gpus or torch.cuda.device_count()
    assert n_gpus >= 2, (
        f"Need >= 2 GPUs, got {torch.cuda.device_count()}. "
        f"Set CUDA_VISIBLE_DEVICES."
    )
    assert n_gpus <= torch.cuda.device_count()

    # Resolve model (downloads if needed)
    print(f"Resolving model: {args.model}")
    model_path, sf_files, index_data = resolve_model(args.model, args.cache_dir)
    index_file = Path(model_path) / "model.safetensors.index.json"

    # Build size-balanced device map
    print(f"Building device map for {n_gpus} GPUs...")
    device_map, total_bytes, gpu_bytes, bytes_per_device = \
        build_balanced_device_map(model_path, index_data, n_gpus)

    n_shards = len(sf_files)
    n_tensors = len(device_map)
    total_gb = total_bytes / 1e9

    # Init CUDA
    for i in range(n_gpus):
        _ = torch.zeros(1, device=f"cuda:{i}")
    sync_gpus()

    gpu_name = torch.cuda.get_device_name(0)
    cache_str = "cold (page cache dropped)" if args.cold_cache else "warm"

    # Test page cache dropping early so we fail fast
    if args.cold_cache:
        try:
            drop_page_cache()
        except Exception as e:
            print(f"ERROR: Cannot drop page cache: {e}", file=sys.stderr)
            print("Ensure passwordless sudo is configured for this command.",
                  file=sys.stderr)
            sys.exit(1)

    print("=" * 80)
    print(f"Weight Loading Benchmark: {args.model}")
    print(f"  {total_gb:.1f} GB, {n_shards} shards, {n_tensors} tensors")
    print(f"  GPUs: {n_gpus}x {gpu_name}, device_map: balanced")
    for i in range(n_gpus):
        print(f"    cuda:{i}: {gpu_bytes[i]/1e9:.1f} GB")
    print(f"  Cache: {cache_str}")
    print(f"  Warmup: {args.warmup}, Runs: {args.runs}")
    print("=" * 80)

    results = {}  # method_name -> (avg_s, best_s)

    # ── Method 1: transformers async (4 threads) ──
    if not args.skip_transformers:
        print(f"\n--- transformers async (4 threads) ---")
        print(f"    (subprocess: {args.transformers_venv})")
        try:
            times = bench_transformers(
                sf_files, device_map, bytes_per_device,
                "async", args.warmup, args.runs, args.cold_cache,
                args.transformers_venv,
            )
            avg_s, best_s, line = fmt_result(times, total_bytes)
            print(line)
            results["transformers_async"] = (avg_s, best_s)
        except Exception as e:
            print(f"  FAILED: {e}")

        # ── Method 2: transformers sync ──
        print(f"\n--- transformers sync ---")
        print(f"    (subprocess: {args.transformers_venv})")
        try:
            times = bench_transformers(
                sf_files, device_map, bytes_per_device,
                "sync", args.warmup, args.runs, args.cold_cache,
                args.transformers_venv,
            )
            avg_s, best_s, line = fmt_result(times, total_bytes)
            print(line)
            results["transformers_sync"] = (avg_s, best_s)
        except Exception as e:
            print(f"  FAILED: {e}")

    # ── Method 3: ours io_uring scatter ──
    print(f"\n--- ours io_uring scatter ---")
    try:
        times = bench_ours(
            index_file, device_map, "io_uring",
            args.warmup, args.runs, args.cold_cache,
        )
        avg_s, best_s, line = fmt_result(times, total_bytes)
        print(line)
        results["ours_iouring"] = (avg_s, best_s)
    except Exception as e:
        print(f"  FAILED: {e}")

    # ── Method 4: ours mmap ──
    print(f"\n--- ours mmap ---")
    try:
        times = bench_ours(
            index_file, device_map, "mmap",
            args.warmup, args.runs, args.cold_cache,
        )
        avg_s, best_s, line = fmt_result(times, total_bytes)
        print(line)
        results["ours_mmap"] = (avg_s, best_s)
    except Exception as e:
        print(f"  FAILED: {e}")

    # ── Summary ──
    print(f"\n{'='*80}")
    print("Summary (sorted by avg time)")
    print(f"{'='*80}")
    if results:
        sorted_res = sorted(results.items(), key=lambda x: x[1][0])
        fastest_avg = sorted_res[0][1][0]
        gb = total_bytes / 1e9
        for name, (avg_s, best_s) in sorted_res:
            ratio = avg_s / fastest_avg
            tag = "  <-- fastest" if ratio == 1.0 else ""
            print(
                f"  {name:<25} avg {avg_s*1000:7.0f}ms  "
                f"best {best_s*1000:7.0f}ms  "
                f"{gb/best_s:5.1f} GB/s  "
                f"({ratio:.2f}x){tag}"
            )

    print(f"\nDone!")


if __name__ == "__main__":
    main()

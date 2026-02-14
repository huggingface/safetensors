"""Tensor Parallelism benchmark for safe_open().

Compares:
  1. Baseline: safe_open(device_map, backend="io_uring") — no TP, full tensors
  2. TP:       safe_open(device_map, tp_plan, backend="io_uring") — TP-sliced tensors

The TP plan matches Qwen2's architecture:
  - colwise: q_proj, k_proj, v_proj, gate_proj, up_proj
  - rowwise: o_proj, down_proj
  - passthrough: layernorms, embed_tokens, lm_head

Usage:
  CUDA_VISIBLE_DEVICES=2,3 LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu \
    ~/.venv/safetensors/bin/python bench_tp.py
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


QWEN2_TP_PLAN = {
    "model.layers.*.self_attn.q_proj": "colwise",
    "model.layers.*.self_attn.k_proj": "colwise",
    "model.layers.*.self_attn.v_proj": "colwise",
    "model.layers.*.self_attn.o_proj": "rowwise",
    "model.layers.*.mlp.gate_proj": "colwise",
    "model.layers.*.mlp.up_proj": "colwise",
    "model.layers.*.mlp.down_proj": "rowwise",
}


def parse_args():
    p = argparse.ArgumentParser(description="TP loading benchmark")
    p.add_argument("--model", default="Qwen/Qwen2.5-14B")
    p.add_argument("--cache-dir", default="/raid/hf_cache")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--backend", default="io_uring", choices=["io_uring", "cufile"])
    return p.parse_args()


def resolve_model(model_id, cache_dir):
    model_path = snapshot_download(
        model_id, cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json"],
        ignore_patterns=["*.bin", "*.pt", "*.onnx"],
    )
    index_file = Path(model_path) / "model.safetensors.index.json"
    with open(index_file) as f:
        idx = json.load(f)
    sf_files = sorted(set(idx["weight_map"].values()))
    sf_files = [Path(model_path) / f for f in sf_files]
    return model_path, index_file, sf_files


def build_device_map(index_file, world_size):
    """Round-robin tensor→device map across world_size GPUs."""
    with open(index_file) as f:
        idx = json.load(f)
    tensor_names = sorted(idx["weight_map"].keys())
    return {name: f"cuda:{i % world_size}" for i, name in enumerate(tensor_names)}


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


def bench_baseline(index_file, device_map, backend, n_warmup, n_runs):
    """Load with device_map only — no TP slicing."""
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


def bench_tp(index_file, device_map, tp_plan, rank, world_size, backend, n_warmup, n_runs):
    """Load with device_map + tp_plan — TP-sliced tensors."""
    tensor_names = sorted(device_map.keys())
    times = []

    for i in range(n_warmup + n_runs):
        clear_gpu_memory()
        sync_gpus()
        t0 = time.perf_counter()

        with safe_open(
            str(index_file), framework="pt", device=device_map, backend=backend,
            tp_plan=tp_plan, tp_rank=rank, tp_world_size=world_size,
        ) as sf:
            for name in tensor_names:
                _ = sf.get_tensor(name)

        sync_gpus()
        elapsed = time.perf_counter() - t0

        if i >= n_warmup:
            times.append(elapsed)

    return times


def compute_tp_data_size(index_file, tp_plan, world_size):
    """Estimate total bytes loaded per rank with TP vs without TP.

    This gives us the theoretical I/O reduction from colwise slicing.
    """
    with open(index_file) as f:
        idx = json.load(f)

    # We need tensor shapes/dtypes from the safetensors files
    from safetensors import safe_open
    with safe_open(str(index_file), framework="pt", device="cpu") as sf:
        total_bytes_no_tp = 0
        total_bytes_tp_rank0 = 0

        for name in sf.keys():
            tensor = sf.get_tensor(name)
            nbytes = tensor.nelement() * tensor.element_size()
            total_bytes_no_tp += nbytes

            # Check if this tensor matches any TP pattern
            matched = False
            for pattern, strategy in tp_plan.items():
                # Simple pattern match for estimation
                stripped = name
                for suffix in (".weight", ".bias"):
                    if stripped.endswith(suffix):
                        stripped = stripped[:-len(suffix)]
                        break

                # Check pattern match (simplified)
                pat_parts = pattern.split("*")
                if len(pat_parts) == 2:
                    if stripped.startswith(pat_parts[0]) and stripped.endswith(pat_parts[1]):
                        middle = stripped[len(pat_parts[0]):len(stripped)-len(pat_parts[1])]
                        if middle.isdigit():
                            matched = True
                            if strategy == "colwise":
                                # Colwise: only load 1/world_size of the bytes
                                tp_bytes = nbytes // world_size
                                # Ceiling division for uneven splits
                                shape = list(tensor.shape)
                                if len(shape) >= 2:
                                    chunk = (shape[-2] + world_size - 1) // world_size
                                    tp_bytes = chunk * (nbytes // shape[-2]) if shape[-2] > 0 else 0
                                elif len(shape) == 1:
                                    chunk = (shape[0] + world_size - 1) // world_size
                                    tp_bytes = chunk * tensor.element_size()
                                total_bytes_tp_rank0 += tp_bytes
                            else:
                                # Rowwise: load full tensor (narrow on GPU)
                                total_bytes_tp_rank0 += nbytes
                            break

            if not matched:
                total_bytes_tp_rank0 += nbytes

    return total_bytes_no_tp, total_bytes_tp_rank0


def main():
    args = parse_args()

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= args.world_size, \
        f"Need >= {args.world_size} GPUs, got {n_gpus}. Set CUDA_VISIBLE_DEVICES"

    print(f"Resolving model: {args.model}")
    model_path, index_file, sf_files = resolve_model(args.model, args.cache_dir)
    model_size = get_model_size_gb(sf_files)

    device_map = build_device_map(index_file, args.world_size)
    n_tensors = len(device_map)

    # Initialize CUDA
    for i in range(n_gpus):
        _ = torch.zeros(1, device=f"cuda:{i}")
    sync_gpus()

    gpu_names = [torch.cuda.get_device_name(i) for i in range(args.world_size)]

    # Compute theoretical I/O reduction
    total_no_tp, total_tp = compute_tp_data_size(index_file, QWEN2_TP_PLAN, args.world_size)

    print("=" * 80)
    print("Tensor Parallelism Loading Benchmark")
    print("=" * 80)
    print(f"  Model:       {args.model} ({model_size:.2f} GiB, {len(sf_files)} shards)")
    print(f"  GPUs:        {', '.join(f'cuda:{i} ({gpu_names[i]})' for i in range(args.world_size))}")
    print(f"  Tensors:     {n_tensors}")
    print(f"  Backend:     {args.backend}")
    print(f"  World size:  {args.world_size}")
    print(f"  Warmup:      {args.warmup}, Runs: {args.runs}")
    print(f"  Data (no TP): {total_no_tp / 1e9:.2f} GB")
    print(f"  Data (TP r0): {total_tp / 1e9:.2f} GB ({100 * total_tp / total_no_tp:.1f}%)")
    print(f"  I/O savings:  {(total_no_tp - total_tp) / 1e9:.2f} GB ({100 * (1 - total_tp / total_no_tp):.1f}%)")
    print("=" * 80)

    results = {}

    # ── Baseline: no TP ──
    print(f"\n--- Baseline: device_map only ({args.backend}) ---")
    times = bench_baseline(index_file, device_map, args.backend, args.warmup, args.runs)
    avg = sum(times) / len(times) * 1000
    best = min(times) * 1000
    ms_list = [int(t * 1000) for t in times]
    bw = model_size / (best / 1000)
    print(f"  avg {avg:7.0f}ms  best {best:7.0f}ms  {bw:5.1f} GiB/s  {ms_list}")
    results["baseline"] = (avg, best)

    # ── TP rank 0 ──
    for rank in range(args.world_size):
        print(f"\n--- TP rank {rank}/{args.world_size} ({args.backend}) ---")
        times = bench_tp(
            index_file, device_map, QWEN2_TP_PLAN, rank, args.world_size,
            args.backend, args.warmup, args.runs,
        )
        avg = sum(times) / len(times) * 1000
        best = min(times) * 1000
        ms_list = [int(t * 1000) for t in times]
        tp_data_gb = total_tp / (1024**3)
        bw = tp_data_gb / (best / 1000)
        print(f"  avg {avg:7.0f}ms  best {best:7.0f}ms  {bw:5.1f} GiB/s (eff)  {ms_list}")
        results[f"tp_rank{rank}"] = (avg, best)

    # ── Summary ──
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    baseline_avg = results["baseline"][0]
    for name, (avg, best) in sorted(results.items(), key=lambda x: x[1][0]):
        speedup = baseline_avg / avg
        tag = ""
        if "tp" in name:
            tag = f"  ({speedup:.2f}x vs baseline)"
        print(f"  {name:<20} avg {avg:7.0f}ms  best {best:7.0f}ms{tag}")

    print(f"\n  Theoretical I/O reduction: {100 * (1 - total_tp / total_no_tp):.1f}%")
    if "tp_rank0" in results:
        actual_speedup = results["baseline"][0] / results["tp_rank0"][0]
        print(f"  Actual speedup (avg):     {actual_speedup:.2f}x")

    print("\nDone!")


if __name__ == "__main__":
    main()

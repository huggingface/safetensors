#!/usr/bin/env python3
"""
Multi-GPU Loading Benchmark: safetensors vs fastsafetensors vs PyTorch

This benchmark compares GPU loading performance using REAL models from HuggingFace:
1. safetensors load_file with device_map (direct multi-GPU loading)
2. safetensors safe_open with mmap backend
3. safetensors safe_open with io_uring backend
4. fastsafetensors (requires torch.distributed)
5. transformers AutoModel (how most users load models)

IMPORTANT: All benchmarks force data materialization with .sum() to ensure
the data is actually transferred to GPU, not just lazily mapped.

Usage:
    # Standard benchmark with phi-2:
    python benchmark_multi_gpu.py

    # With a different model:
    python benchmark_multi_gpu.py --model meta-llama/Llama-3.2-3B

    # With fastsafetensors (requires torchrun):
    torchrun --nproc_per_node=4 benchmark_multi_gpu.py --fastsafetensors
"""

import argparse
import gc
import os
import time
from pathlib import Path


def _filter_available_gpus():
    """Probe GPUs and set CUDA_VISIBLE_DEVICES to skip busy ones.

    Must be called BEFORE any torch.cuda operations to avoid tainting
    the CUDA context with an error from a busy device.
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return  # User already specified devices

    import subprocess
    import sys

    # Use a single subprocess to probe all GPUs at once
    probe_script = """
import torch, json
results = []
for i in range(torch.cuda.device_count()):
    try:
        torch.cuda.mem_get_info(i)
        results.append(i)
    except Exception:
        pass
print(json.dumps(results))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe_script],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return  # Can't probe, let it fail normally

    import json

    try:
        available = json.loads(result.stdout.strip())
    except (json.JSONDecodeError, ValueError):
        return

    # Get total count from the probe subprocess
    total_result = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
        capture_output=True,
        text=True,
    )
    total = int(total_result.stdout.strip()) if total_result.returncode == 0 else len(available)

    if len(available) < total:
        busy = set(range(total)) - set(available)
        for g in sorted(busy):
            print(f"  GPU {g}: busy (skipping)")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in available)
        print(
            f"  Using {len(available)}/{total} GPUs: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"
        )


_filter_available_gpus()

import torch


def sync_all_gpus(num_gpus):
    """Synchronize all GPUs."""
    for i in range(num_gpus):
        torch.cuda.synchronize(i)


def materialize_tensors(tensors_dict):
    """Force materialization of all tensors by computing their sum.

    This ensures data is actually transferred to GPU, not lazily mapped.
    Returns checksum for verification.
    """
    total = 0.0
    for name, t in tensors_dict.items():
        total += float(t.sum())
    return total


def get_model_files(model_id, cache_dir=None):
    """Download model and return paths to safetensors files."""
    from huggingface_hub import snapshot_download

    print(f"Downloading {model_id}...")
    model_path = snapshot_download(
        model_id,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json"],
        ignore_patterns=["*.bin", "*.pt", "*.onnx"],
    )
    print(f"Model path: {model_path}")

    # Find safetensors files
    sf_files = list(Path(model_path).glob("*.safetensors"))
    if not sf_files:
        raise ValueError(f"No safetensors files found in {model_path}")

    # Check for sharded model
    index_file = Path(model_path) / "model.safetensors.index.json"
    if index_file.exists():
        import json

        with open(index_file) as f:
            index = json.load(f)
        # Get unique shard files
        shard_files = set(index["weight_map"].values())
        sf_files = [Path(model_path) / f for f in sorted(shard_files)]
        print(f"Sharded model: {len(sf_files)} files")
    else:
        print("Single file model")

    return model_path, sf_files


def create_device_map_from_model(model_path, num_gpus):
    """Create a device map distributing layers across GPUs."""
    from safetensors import safe_open

    # Get all tensor names
    sf_files = list(Path(model_path).glob("*.safetensors"))
    tensor_names = []
    for sf_file in sf_files:
        with safe_open(sf_file, framework="pt") as sf:
            tensor_names.extend(sf.keys())

    # Create device map
    device_map = {}

    # Find layer indices
    layer_indices = set()
    for name in tensor_names:
        if ".layers." in name:
            # Extract layer number
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_indices.add(int(parts[i + 1]))
                    except ValueError:
                        pass

    layer_indices = sorted(layer_indices)
    num_layers = len(layer_indices)
    print(f"Model has {num_layers} layers, distributing across {num_gpus} GPUs")

    # Assign layers to GPUs (round-robin)
    layer_to_gpu = {}
    for i, layer_idx in enumerate(layer_indices):
        layer_to_gpu[layer_idx] = i % num_gpus

    # Build device map
    for name in tensor_names:
        assigned = False
        if ".layers." in name:
            parts = name.split(".")
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        device_map[name] = f"cuda:{layer_to_gpu[layer_idx]}"
                        assigned = True
                        break
                    except ValueError:
                        pass

        if not assigned:
            # Embeddings go to first GPU, output layers to last
            if "embed" in name.lower():
                device_map[name] = "cuda:0"
            elif "lm_head" in name.lower() or "output" in name.lower():
                device_map[name] = f"cuda:{num_gpus - 1}"
            else:
                device_map[name] = "cuda:0"

    return device_map


def benchmark_safetensors_load_file(sf_files, device_map, num_gpus, warmup=1, runs=3):
    """Benchmark safetensors load_file with device_map."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    # Handle single-device mode (string device_map)
    is_single_device = isinstance(device_map, str)

    # Warmup
    for _ in range(warmup):
        result = {}
        for sf_file in sf_files:
            if is_single_device:
                result.update(load_file(str(sf_file), device=device_map))
            else:
                with safe_open(sf_file, framework="pt") as sf:
                    file_keys = set(sf.keys())
                file_device_map = {k: v for k, v in device_map.items() if k in file_keys}
                result.update(load_file(str(sf_file), device=file_device_map))
        materialize_tensors(result)
        sync_all_gpus(num_gpus)
        del result
        torch.cuda.empty_cache()
        gc.collect()

    # Benchmark
    times = []
    for _ in range(runs):
        sync_all_gpus(num_gpus)
        torch.cuda.empty_cache()
        gc.collect()

        start = time.perf_counter()
        result = {}
        for sf_file in sf_files:
            if is_single_device:
                result.update(load_file(str(sf_file), device=device_map))
            else:
                with safe_open(sf_file, framework="pt") as sf:
                    file_keys = set(sf.keys())
                file_device_map = {k: v for k, v in device_map.items() if k in file_keys}
                result.update(load_file(str(sf_file), device=file_device_map))
        # Force materialization
        materialize_tensors(result)
        sync_all_gpus(num_gpus)
        times.append(time.perf_counter() - start)
        del result

    return times


def _group_tensors_by_device(sf_files, device_map):
    """Group tensor names by target device, keyed per file."""
    from safetensors import safe_open
    from collections import defaultdict

    # file_path -> {device -> [tensor_names]}
    per_file = {}
    for sf_file in sf_files:
        groups = defaultdict(list)
        with safe_open(str(sf_file), framework="pt") as sf:
            for name in sf.keys():
                dev = device_map.get(name, "cuda:0")
                groups[dev].append(name)
        per_file[str(sf_file)] = dict(groups)
    return per_file


def _load_for_device(sf_file, device, names, backend, result):
    """Load tensors for a single device (runs in thread)."""
    from safetensors import safe_open

    with safe_open(sf_file, framework="pt", device=device, backend=backend) as sf:
        for name in names:
            result[name] = sf.get_tensor(name)


def benchmark_safetensors_safe_open(
    sf_files, device_map, num_gpus, backend="mmap", warmup=1, runs=3
):
    """Benchmark safetensors safe_open with specified backend.

    Multi-device: passes device_map dict directly to safe_open.
    The Rust backend calls cudaSetDevice per-tensor for direct GPU loading.
    """
    from safetensors import safe_open

    def run_once():
        result = {}
        for sf_file in sf_files:
            with safe_open(
                str(sf_file), framework="pt", device=device_map, backend=backend
            ) as sf:
                for name in sf.keys():
                    result[name] = sf.get_tensor(name)
        return result

    # Warmup
    for _ in range(warmup):
        result = run_once()
        materialize_tensors(result)
        sync_all_gpus(num_gpus)
        del result
        torch.cuda.empty_cache()
        gc.collect()

    # Benchmark
    times = []
    for _ in range(runs):
        sync_all_gpus(num_gpus)
        torch.cuda.empty_cache()
        gc.collect()

        start = time.perf_counter()
        result = run_once()
        materialize_tensors(result)
        sync_all_gpus(num_gpus)
        times.append(time.perf_counter() - start)
        del result

    return times


def benchmark_safetensors_batch(
    sf_files, device_map, num_gpus, backend="io_uring", warmup=1, runs=3
):
    """Benchmark safetensors get_tensors batch loading."""
    from safetensors import safe_open

    def run_once():
        result = {}
        for sf_file in sf_files:
            with safe_open(
                str(sf_file), framework="pt", device=device_map, backend=backend
            ) as sf:
                names = list(sf.keys())
                result.update(sf.get_tensors(names))
        return result

    # Warmup
    for _ in range(warmup):
        result = run_once()
        materialize_tensors(result)
        sync_all_gpus(num_gpus)
        del result
        torch.cuda.empty_cache()
        gc.collect()

    # Benchmark
    times = []
    for _ in range(runs):
        sync_all_gpus(num_gpus)
        torch.cuda.empty_cache()
        gc.collect()

        start = time.perf_counter()
        result = run_once()
        materialize_tensors(result)
        sync_all_gpus(num_gpus)
        times.append(time.perf_counter() - start)
        del result

    return times


def benchmark_safetensors_iter_tensors(
    sf_files, device_map, num_gpus, backend="mmap", prefetch=4, warmup=1, runs=3
):
    """Benchmark safetensors iter_tensors with specified backend."""
    from safetensors import safe_open

    # For single-device mode, use string device
    if isinstance(device_map, str):
        device = device_map
    else:
        device = None

    # Warmup
    for _ in range(warmup):
        result = {}
        for sf_file in sf_files:
            if device:
                with safe_open(
                    str(sf_file), framework="pt", device=device, backend=backend
                ) as sf:
                    for name, tensor in sf.iter_tensors(prefetch=prefetch):
                        result[name] = tensor
            else:
                # Multi-device: load to CPU then move
                with safe_open(str(sf_file), framework="pt", backend=backend) as sf:
                    for name, tensor in sf.iter_tensors(prefetch=prefetch):
                        target_device = device_map.get(name, "cuda:0")
                        result[name] = tensor.to(target_device)
        materialize_tensors(result)
        sync_all_gpus(num_gpus)
        del result
        torch.cuda.empty_cache()
        gc.collect()

    # Benchmark
    times = []
    for _ in range(runs):
        sync_all_gpus(num_gpus)
        torch.cuda.empty_cache()
        gc.collect()

        start = time.perf_counter()
        result = {}
        for sf_file in sf_files:
            if device:
                with safe_open(
                    str(sf_file), framework="pt", device=device, backend=backend
                ) as sf:
                    for name, tensor in sf.iter_tensors(prefetch=prefetch):
                        result[name] = tensor
            else:
                # Multi-device: load to CPU then move
                with safe_open(str(sf_file), framework="pt", backend=backend) as sf:
                    for name, tensor in sf.iter_tensors(prefetch=prefetch):
                        target_device = device_map.get(name, "cuda:0")
                        result[name] = tensor.to(target_device)
        # Force materialization
        materialize_tensors(result)
        sync_all_gpus(num_gpus)
        times.append(time.perf_counter() - start)
        del result

    return times


def benchmark_transformers_auto(model_id, device_map, num_gpus, warmup=1, runs=3,
                                cache_dir=None, transformers_venv=None):
    """Benchmark transformers AutoModel loading.

    Uses a separate venv with official safetensors to avoid compatibility issues
    with our modified safetensors.
    """
    import json
    import subprocess

    venv = transformers_venv or os.path.expanduser("~/.venv/transformers")
    python = os.path.join(venv, "bin", "python")
    if not os.path.exists(python):
        print(f"  [SKIP] Transformers venv not found at {venv}")
        return None

    # Build a small script that runs the benchmark in the separate venv
    script = f"""
import gc, time, json, os, torch
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
LD = "{venv}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
if LD not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = LD + ":" + os.environ.get("LD_LIBRARY_PATH", "")

from transformers import AutoModelForCausalLM
num_gpus = {num_gpus}
device_map = {json.dumps(device_map) if isinstance(device_map, (dict, str)) else repr(device_map)}
cache_dir = {json.dumps(cache_dir)}

def sync_all():
    for i in range(num_gpus):
        torch.cuda.synchronize(i)

# Warmup
for _ in range({warmup}):
    model = AutoModelForCausalLM.from_pretrained(
        "{model_id}", device_map=device_map, dtype=torch.float16,
        cache_dir=cache_dir,
    )
    sync_all()
    del model
    torch.cuda.empty_cache()
    gc.collect()

# Benchmark
times = []
for _ in range({runs}):
    sync_all()
    torch.cuda.empty_cache()
    gc.collect()
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        "{model_id}", device_map=device_map, dtype=torch.float16,
        cache_dir=cache_dir,
    )
    sync_all()
    times.append(time.perf_counter() - start)
    del model

print(json.dumps(times))
"""
    # Set LD_LIBRARY_PATH for CUDA 12.8 runtime
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

    # Parse the last line as JSON (times array)
    for line in reversed(result.stdout.strip().split("\n")):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def benchmark_fastsafetensors(model_path, num_gpus, warmup=1, runs=3, use_gds=True):
    """Benchmark fastsafetensors with GPUDirect Storage (requires torch.distributed)."""
    try:
        import torch.distributed as dist
        from fastsafetensors import SafeTensorsFileLoader

        if not dist.is_initialized():
            print("  [SKIP] Requires torch.distributed")
            print(
                "  Run with: torchrun --nproc_per_node=4 benchmark_multi_gpu.py --fastsafetensors"
            )
            return None

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        sf_files = sorted(Path(model_path).glob("*.safetensors"))

        # Create loader with GPUDirect enabled (nogds=False)
        loader = SafeTensorsFileLoader(
            pg=dist.group.WORLD,
            device=f"cuda:{rank}",
            nogds=not use_gds,  # nogds=False means GDS enabled
            debug_log=False,  # Disable debug logging
        )

        # Add all files - each rank loads all files, then shards
        # fastsafetensors expects {rank: [filenames]} mapping
        filenames_dict = {r: [str(f) for f in sf_files] for r in range(world_size)}
        loader.add_filenames(filenames_dict)

        # Warmup
        for _ in range(warmup):
            # copy_files_to_device does the actual GDS read
            buffer = loader.copy_files_to_device()
            # Get all tensors
            result = {}
            for key in loader.get_keys():
                result[key] = buffer.get_tensor(key)
            materialize_tensors(result)
            dist.barrier()
            buffer.close()
            del result
            torch.cuda.empty_cache()
            gc.collect()

        # Benchmark
        times = []
        for _ in range(runs):
            dist.barrier()
            torch.cuda.empty_cache()
            gc.collect()

            start = time.perf_counter()
            # copy_files_to_device does the actual GDS read
            buffer = loader.copy_files_to_device()
            # Get all tensors
            result = {}
            for key in loader.get_keys():
                result[key] = buffer.get_tensor(key)
            materialize_tensors(result)
            dist.barrier()
            times.append(time.perf_counter() - start)
            buffer.close()
            del result

        loader.close()
        return times

    except ImportError:
        print("  [SKIP] fastsafetensors not installed")
        return None
    except Exception as e:
        import traceback
        if dist.is_initialized() and dist.get_rank() == 0:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Loading Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/phi-2",
        help="HuggingFace model ID (default: microsoft/phi-2)",
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument(
        "--fastsafetensors",
        action="store_true",
        help="Include fastsafetensors benchmark",
    )
    parser.add_argument(
        "--skip-transformers",
        action="store_true",
        help="Skip transformers benchmark (slow)",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None, help="HuggingFace cache directory"
    )
    parser.add_argument(
        "--single-gpu", action="store_true", help="Force single GPU mode (cuda:0 only)"
    )
    args = parser.parse_args()

    # Initialize torch.distributed if running with torchrun
    import torch.distributed as dist
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        is_main = rank == 0
    else:
        is_main = True
        rank = 0

    if is_main:
        print("=" * 70)
        print("Multi-GPU Loading Benchmark (Real Model)")
        print("=" * 70)
        print()
        print("NOTE: All benchmarks force data materialization with .sum()")
        print("      to ensure data is actually transferred to GPU.")
        print()

    # Check GPUs
    num_gpus = torch.cuda.device_count()
    if is_main:
        print(f"Available GPUs: {num_gpus}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            free_mem = torch.cuda.mem_get_info(i)[0] // 1024 // 1024
            print(
                f"  GPU {i}: {props.name} ({props.total_memory // 1024 // 1024} MB, {free_mem} MB free)"
            )
        print()

    if num_gpus < 1:
        if is_main:
            print("ERROR: Need at least 1 GPU")
        return

    # Download model (only on main rank to avoid race conditions)
    if is_main:
        model_path, sf_files = get_model_files(args.model, cache_dir=args.cache_dir)
    else:
        # Other ranks wait and get the same path
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(
            args.model,
            cache_dir=args.cache_dir,
            allow_patterns=["*.safetensors", "*.json"],
            ignore_patterns=["*.bin", "*.pt", "*.onnx"],
        )
        sf_files = sorted(Path(model_path).glob("*.safetensors"))

    if dist.is_initialized():
        dist.barrier()

    # Calculate total size
    total_bytes = sum(os.path.getsize(f) for f in sf_files)
    if is_main:
        print(f"Total model size: {total_bytes / 1024 / 1024 / 1024:.2f} GB")
        print()

    # Create device map
    if num_gpus > 1 and not args.single_gpu:
        device_map = create_device_map_from_model(model_path, num_gpus)
        if is_main:
            # Count tensors per GPU
            gpu_counts = {f"cuda:{i}": 0 for i in range(num_gpus)}
            for device in device_map.values():
                gpu_counts[device] += 1
            print("Tensors per GPU:")
            for device, count in sorted(gpu_counts.items()):
                print(f"  {device}: {count}")
    else:
        device_map = "cuda:0"
        if is_main:
            print("Single GPU mode")
    if is_main:
        print()

    results = {}

    # Only run safetensors benchmarks on main rank (they don't use distributed)
    if is_main:
        # Benchmark 1: safe_open with mmap backend (direct multi-GPU)
        print("1. safetensors mmap (direct multi-GPU):")
        try:
            times = benchmark_safetensors_safe_open(
                sf_files, device_map, num_gpus, backend="mmap", runs=args.runs
            )
            avg_time = sum(times) / len(times)
            throughput = total_bytes / avg_time / 1024 / 1024 / 1024
            print(f"   {avg_time * 1000:.1f}ms ({throughput:.2f} GB/s)")
            results["mmap"] = avg_time * 1000
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"   [ERROR] {e}")

        # Benchmark 2: safe_open with io_uring backend (direct multi-GPU)
        print("2. safetensors io_uring (direct multi-GPU):")
        try:
            times = benchmark_safetensors_safe_open(
                sf_files, device_map, num_gpus, backend="io_uring", runs=args.runs
            )
            avg_time = sum(times) / len(times)
            throughput = total_bytes / avg_time / 1024 / 1024 / 1024
            print(f"   {avg_time * 1000:.1f}ms ({throughput:.2f} GB/s)")
            results["io_uring"] = avg_time * 1000
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"   [ERROR] {e}")

        # Benchmark 3: io_uring batch loading with get_tensors
        print("3. safetensors io_uring batch (get_tensors):")
        try:
            times = benchmark_safetensors_batch(
                sf_files, device_map, num_gpus, backend="io_uring", runs=args.runs
            )
            avg_time = sum(times) / len(times)
            throughput = total_bytes / avg_time / 1024 / 1024 / 1024
            print(f"   {avg_time * 1000:.1f}ms ({throughput:.2f} GB/s)")
            results["io_uring_batch"] = avg_time * 1000
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"   [ERROR] {e}")

        # Benchmark 4: cuFile/GDS backend (direct NVMe->GPU DMA)
        print("4. safetensors cuFile/GDS (direct NVMe->GPU):")
        try:
            times = benchmark_safetensors_safe_open(
                sf_files, device_map, num_gpus, backend="cufile", runs=args.runs
            )
            avg_time = sum(times) / len(times)
            throughput = total_bytes / avg_time / 1024 / 1024 / 1024
            print(f"   {avg_time * 1000:.1f}ms ({throughput:.2f} GB/s)")
            results["cufile"] = avg_time * 1000
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"   [ERROR] {e}")

    # Synchronize before fastsafetensors benchmark
    if dist.is_initialized():
        dist.barrier()

    # Benchmark 6: transformers AutoModel (only on main rank)
    if not args.skip_transformers and is_main:
        print("6. transformers AutoModelForCausalLM.from_pretrained():")
        try:
            # transformers uses "auto" or dict device_map
            if num_gpus > 1 and not args.single_gpu:
                tf_device_map = "auto"
            else:
                tf_device_map = {"": "cuda:0"}
            times = benchmark_transformers_auto(
                args.model, tf_device_map, num_gpus, runs=args.runs,
                cache_dir=args.cache_dir,
            )
            if times is None:
                raise RuntimeError("Transformers benchmark returned no results")
            avg_time = sum(times) / len(times)
            throughput = total_bytes / avg_time / 1024 / 1024 / 1024
            print(f"   {avg_time * 1000:.1f}ms ({throughput:.2f} GB/s)")
            results["transformers"] = avg_time * 1000
        except Exception as e:
            print(f"   [ERROR] {e}")

    # Benchmark 7: fastsafetensors (runs on all ranks, uses GPUDirect)
    if args.fastsafetensors:
        if is_main:
            print("7. fastsafetensors (GPUDirect):")
        times = benchmark_fastsafetensors(model_path, num_gpus, runs=args.runs)
        if times and is_main:
            avg_time = sum(times) / len(times)
            throughput = total_bytes / avg_time / 1024 / 1024 / 1024
            print(f"   {avg_time * 1000:.1f}ms ({throughput:.2f} GB/s)")
            results["fastsafetensors_gds"] = avg_time * 1000

    # Summary
    if is_main:
        print()
        print("=" * 70)
        print("Summary")
        print("=" * 70)
        if results:
            baseline = min(results.values())
            for name, time_ms in sorted(results.items(), key=lambda x: x[1]):
                ratio = time_ms / baseline
                marker = " (fastest)" if time_ms == baseline else ""
                throughput = total_bytes / (time_ms / 1000) / 1024 / 1024 / 1024
                print(
                    f"{name:25s}: {time_ms:8.1f}ms  {throughput:6.2f} GB/s  ({ratio:.2f}x){marker}"
                )

        print()
        print("Done!")

    # Cleanup distributed
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

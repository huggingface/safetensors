import os
import tempfile
import time
from typing import Dict, List, Tuple

import torch
from safetensors.torch import save_file, safe_open

RUN_COUNT = 10
RUN_WARMUP = 0


def create_test_file(size_mb: int) -> Tuple[str, Dict[str, torch.Tensor]]:
    """
    Create a temporary safetensors file with a tensor of specified size.
    
    Args:
        size_mb: Size of the tensor in megabytes
        
    Returns:
        Tuple of (filename, tensors_dict)
    """
    # Calculate shape for float32 tensor (4 bytes per element)
    num_elements = (size_mb * 1024 * 1024) // 4
    # Use a 2D tensor with reasonable dimensions
    rows = 1000
    cols = num_elements // rows
    
    tensor = torch.randn(rows, cols, dtype=torch.float32)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors")
    temp_file.close()
    
    save_file({"data": tensor}, temp_file.name)
    
    return temp_file.name, {"data": tensor}


def benchmark_standard_loading(filename: str, tensor_name: str, warmup: int = RUN_WARMUP, iterations: int = RUN_COUNT) -> float:
    """
    Benchmark standard memory-mapped loading + GPU transfer.
    
    Args:
        filename: Path to safetensors file
        tensor_name: Name of tensor to load
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations
        
    Returns:
        Average loading time in seconds
    """
    # Warmup
    for _ in range(warmup):
        with safe_open(filename, framework="pt", device="cuda:0", use_gds=False) as f:
            tensor = f.get_tensor(tensor_name)
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        with safe_open(filename, framework="pt", device="cuda:0", use_gds=False) as f:
            tensor = f.get_tensor(tensor_name)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    return sum(times) / len(times)


def benchmark_gds_loading(filename: str, tensor_name: str, warmup: int = RUN_WARMUP, iterations: int = RUN_COUNT) -> float:
    """
    Benchmark GDS direct NVMe to GPU loading.
    
    Args:
        filename: Path to safetensors file
        tensor_name: Name of tensor to load
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations
        
    Returns:
        Average loading time in seconds
    """
    # Warmup
    for _ in range(warmup):
        with safe_open(filename, framework="pt", device="cuda:0", use_gds=True) as f:
            tensor = f.get_tensor(tensor_name)
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        with safe_open(filename, framework="pt", device="cuda:0", use_gds=True) as f:
            tensor = f.get_tensor(tensor_name)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)

    return sum(times) / len(times)


def run_benchmark_suite():
    """Run comprehensive benchmark suite."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmarks")
        return
    
    # Test different sizes
    # sizes_mb = [10, 50, 100] # Sizes in MB
    sizes_mb = [10, 50, 100, 250, 500, 1000, 2000, 3000] # Sizes in MB
    
    results = []
    
    for size_mb in sizes_mb:
        print(f"\nBenchmarking {size_mb} MB tensor...")
        
        # Create test file
        filename, tensors = create_test_file(size_mb)
        
        try:
            # Benchmark standard loading
            print(" - Standard (mmap + GPU transfer)...", end=" ", flush=True)
            time_standard = benchmark_standard_loading(filename, "data")
            throughput_standard = size_mb / time_standard
            print(f"{time_standard*1000:.2f} ms ({throughput_standard:.1f} MB/s)")
            
            # Benchmark GDS loading
            print(" - GDS (NVMe + GPU)...", end=" ", flush=True)
            time_gds = benchmark_gds_loading(filename, "data")
            throughput_gds = size_mb / time_gds
            print(f"{time_gds*1000:.2f} ms ({throughput_gds:.1f} MB/s)")
            
            # Calculate speedup
            speedup = time_standard / time_gds
            throughput_improvement = ((throughput_gds - throughput_standard) / throughput_standard) * 100
            
            print()
            if speedup > 1:
                print(f"  GDS is {speedup:.2f}x faster!")
                print(f"  Throughput improvement: {throughput_improvement:.1f}%")
            elif speedup < 1:
                print(f"  GDS is {1/speedup:.2f}x slower")
                print(f"  Throughput reduction: {throughput_improvement:.1f}%")
            else:
                print(f"  Similar performance")
            
            results.append({
                "size_mb": size_mb,
                "time_standard": time_standard,
                "time_gds": time_gds,
                "speedup": speedup,
                "throughput_standard": throughput_standard,
                "throughput_gds": throughput_gds,
            })
            
        finally:
            # Clean up
            if os.path.exists(filename):
                os.unlink(filename)
    
    # Print summary
    print()
    print("Summary")
    print()
    print(f"{'Size (MB)':<12} {'Standard (ms)':<18} {'GDS (ms)':<18} {'Speedup':<12} {'GDS Throughput'}")
    print("-" * 80)
    
    for r in results:
        size_mb = r["size_mb"]
        time_std = r["time_standard"] * 1000
        time_gds = r["time_gds"] * 1000
        speedup = r["speedup"]
        throughput = r["throughput_gds"]
        
        speedup_str = f"{speedup:.2f}x" if speedup >= 1 else f"{1/speedup:.2f}x slower"
        
        print(f"{size_mb:<12} {time_std:<18.2f} {time_gds:<18.2f} {speedup_str:<12} {throughput:.1f} MB/s")
    
    print()
    
    # Overall statistics
    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    max_speedup = max(r["speedup"] for r in results)
    max_speedup_size = [r["size_mb"] for r in results if r["speedup"] == max_speedup][0]
    
    print()
    print("Findings:")
    print(f" - Average speedup: {avg_speedup:.2f}x")
    print(f" - Maximum speedup: {max_speedup:.2f}x (at {max_speedup_size} MB)")
    print(f" - Best throughput: {max(r['throughput_gds'] for r in results):.1f} MB/s")
    # top 10 speedup for size
    top_speedups = sorted(results, key=lambda r: r["speedup"], reverse=True)[:10]
    print(" - Top 3 speedups by size:")
    for r in top_speedups:
        print(f"    - {r['size_mb']} MB: {r['speedup']:.2f}x")
    # worst 3 speedup for size
    worst_speedups = sorted(results, key=lambda r: r["speedup"])[:3]
    print(" - Worst 3 speedups by size:")
    for r in worst_speedups:
        print(f"    - {r['size_mb']} MB: {r['speedup']:.2f}x")
    print()


if __name__ == "__main__":
    run_benchmark_suite()

"""
Minimal reproduction script for Windows safetensors segfault.
Tests both safetensors.safe_open and numpy memmap patterns.

Run with: python test_windows_repro.py
"""

import gc
import sys
import torch
from pathlib import Path
from safetensors.torch import save_file
from safetensors import safe_open

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Platform: {sys.platform}")
print()

# Create test file with multiple tensors (simulating model shards)
# Using larger sizes to create memory pressure
TEST_FILE = Path("./test_repro.safetensors")
NUM_TENSORS = 100
TENSOR_SIZE = (2048, 2048)  # ~16MB per tensor, ~1.6GB total

print(f"Creating test file with {NUM_TENSORS} tensors of shape {TENSOR_SIZE}...")
tensors = {f"layer_{i}.weight": torch.randn(TENSOR_SIZE) for i in range(NUM_TENSORS)}
save_file(tensors, str(TEST_FILE))
print(f"File size: {TEST_FILE.stat().st_size / 1024 / 1024:.1f} MB")
print()


def test_safetensors_safe_open():
    """Test using safetensors.safe_open directly."""
    print("=" * 60)
    print("TEST: safetensors.safe_open pattern")
    print("=" * 60)

    # Pattern 1: Load all tensors, access after context exit
    print("\n1a. Load tensors, access after context exit...")
    loaded = {}
    with safe_open(str(TEST_FILE), framework="pt") as f:
        for key in f.keys():
            loaded[key] = f.get_tensor(key)

    gc.collect()
    gc.collect()

    for key, tensor in loaded.items():
        val = tensor[0, 0].item()
    print(f"    Accessed {len(loaded)} tensors - OK")

    # Pattern 2: Lazy loading with lambdas (like convert_hf_to_gguf.py)
    print("\n1b. Lazy loading with lambdas...")
    lazy_tensors = {}
    with safe_open(str(TEST_FILE), framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            lazy_tensors[key] = lambda t=tensor: t

    gc.collect()
    gc.collect()
    gc.collect()

    for key, get_tensor in lazy_tensors.items():
        t = get_tensor()
        val = t[0, 0].item()
    print(f"    Lazy accessed {len(lazy_tensors)} tensors - OK")

    # Pattern 3: Repeated open/close cycles
    print("\n1c. Repeated open/close cycles (100x)...")
    for i in range(100):
        with safe_open(str(TEST_FILE), framework="pt") as f:
            t = f.get_tensor("layer_0.weight")
        gc.collect()
        _ = t.sum().item()
    print("    100 cycles completed - OK")

    print("\nsafetensors.safe_open tests PASSED")


def test_numpy_framework():
    """Test using safetensors with framework='numpy'."""
    print("\n" + "=" * 60)
    print("TEST: safetensors.safe_open with framework='numpy'")
    print("=" * 60)

    # Pattern 1: Load all tensors, access after context exit
    print("\n2a. Load numpy arrays, access after context exit...")
    loaded = {}
    with safe_open(str(TEST_FILE), framework="numpy") as f:
        for key in f.keys():
            loaded[key] = f.get_tensor(key)

    gc.collect()
    gc.collect()

    for key, arr in loaded.items():
        val = arr[0, 0].item()
    print(f"    Accessed {len(loaded)} numpy arrays - OK")

    # Pattern 2: Lazy loading with lambdas
    print("\n2b. Lazy loading numpy arrays with lambdas...")
    lazy_tensors = {}
    with safe_open(str(TEST_FILE), framework="numpy") as f:
        for key in f.keys():
            arr = f.get_tensor(key)
            lazy_tensors[key] = lambda a=arr: a

    gc.collect()
    gc.collect()
    gc.collect()

    for key, get_arr in lazy_tensors.items():
        arr = get_arr()
        val = arr[0, 0].item()
    print(f"    Lazy accessed {len(lazy_tensors)} numpy arrays - OK")

    # Pattern 3: Convert numpy to torch after context exit
    print("\n2c. Load numpy, convert to torch after context exit...")
    loaded_np = {}
    with safe_open(str(TEST_FILE), framework="numpy") as f:
        for key in f.keys():
            loaded_np[key] = f.get_tensor(key)

    gc.collect()
    gc.collect()

    # Convert to torch tensors after context is closed
    for key, arr in loaded_np.items():
        tensor = torch.from_numpy(arr)
        val = tensor[0, 0].item()
    print(f"    Converted and accessed {len(loaded_np)} tensors - OK")

    # Pattern 4: Repeated open/close cycles
    print("\n2d. Repeated open/close cycles with numpy (100x)...")
    for i in range(100):
        with safe_open(str(TEST_FILE), framework="numpy") as f:
            arr = f.get_tensor("layer_0.weight")
        gc.collect()
        _ = arr.sum().item()
    print("    100 cycles completed - OK")

    print("\nsafetensors numpy framework tests PASSED")


def test_mixed_access_patterns():
    """Test patterns that might stress the system more."""
    print("\n" + "=" * 60)
    print("TEST: Mixed/stress patterns")
    print("=" * 60)

    print("\n3a. Interleaved load and access...")
    tensors = []
    with safe_open(str(TEST_FILE), framework="pt") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            tensors.append(t)
            # Access while still in context
            _ = t.mean().item()

    # Access again after context exit
    gc.collect()
    for t in tensors:
        _ = t.sum().item()
    print("    Interleaved access - OK")

    print("\n3b. Multiple shards like 70B model (30 shards, ~7GB total)...")
    # Simulate 70B model: 30 shards with many tensors each
    NUM_SHARDS = 30
    TENSORS_PER_SHARD = 30
    SHARD_TENSOR_SIZE = (2048, 1024)  # ~8MB per tensor, ~240MB per shard

    print(f"    Creating {NUM_SHARDS} shards...")
    for i in range(NUM_SHARDS):
        fname = f"./test_shard_{i}.safetensors"
        save_file(
            {
                f"layer_{j}.weight": torch.randn(SHARD_TENSOR_SIZE)
                for j in range(TENSORS_PER_SHARD)
            },
            fname,
        )
        if (i + 1) % 10 == 0:
            print(f"    Created {i + 1}/{NUM_SHARDS} shards...")

    print(f"    Loading tensors from all shards...")
    all_tensors = []
    for i in range(NUM_SHARDS):
        fname = f"./test_shard_{i}.safetensors"
        with safe_open(fname, framework="pt") as f:
            for key in f.keys():
                all_tensors.append(f.get_tensor(key))
        gc.collect()  # GC after each shard like gguf does

    print(f"    Loaded {len(all_tensors)} tensors, accessing after context exit...")
    # Access all tensors after all files closed - this is where crash might happen
    for idx, t in enumerate(all_tensors):
        _ = t[0, 0].item()
        if (idx + 1) % 200 == 0:
            print(f"    Accessed {idx + 1}/{len(all_tensors)} tensors...")
    print(f"    Accessed {len(all_tensors)} tensors from {NUM_SHARDS} shards - OK")

    # Cleanup shards
    for i in range(NUM_SHARDS):
        Path(f"./test_shard_{i}.safetensors").unlink()

    print("\n3c. Heavy .item() access pattern (matches crash stacktrace)...")
    # The original crash was in _local_scalar_dense which is the .item() path
    with safe_open(str(TEST_FILE), framework="pt") as f:
        tensors = {k: f.get_tensor(k) for k in f.keys()}

    gc.collect()
    gc.collect()

    # Hammer .item() calls like the conversion script might
    count = 0
    for name, t in tensors.items():
        for i in range(min(10, t.shape[0])):
            for j in range(min(10, t.shape[1])):
                _ = t[i, j].item()
                count += 1
    print(f"    {count} .item() calls completed - OK")

    print("\nMixed pattern tests PASSED")


if __name__ == "__main__":
    try:
        test_safetensors_safe_open()
        test_numpy_framework()
        test_mixed_access_patterns()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
    finally:
        # Cleanup
        if TEST_FILE.exists():
            TEST_FILE.unlink()

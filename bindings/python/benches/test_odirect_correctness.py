#!/usr/bin/env python3
"""
O_DIRECT Data Correctness Validation Test

Verifies that data loaded via io_uring (with O_DIRECT + alignment) exactly
matches data loaded via mmap (ground truth). This catches subtle bugs like:
- Alignment padding not stripped correctly
- Off-by-one in aligned offset rounding
- Buffer overflow from aligned reads exceeding staging buffer
- Incorrect useful_len tracking across chunk boundaries
- Cross-device stream issues with cudaMemcpyAsync

Tests both single-tensor and multi-tensor loading across various sizes
and alignment scenarios.
"""

import argparse
import os
import sys
import struct
import tempfile
from pathlib import Path

import torch

# Ensure correct CUDA runtime
cuda_lib = os.path.expanduser(
    "~/.venv/safetensors/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
)
if os.path.exists(cuda_lib):
    os.environ["LD_LIBRARY_PATH"] = cuda_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")


def create_test_safetensors(path, tensors):
    """Create a safetensors file with given tensors.

    This manually writes the safetensors binary format to ensure precise
    control over tensor offsets (important for alignment testing).
    """
    import json

    # Build header metadata
    header = {}
    offset = 0
    tensor_data_parts = []

    dtype_map = {
        torch.float16: "F16",
        torch.float32: "F32",
        torch.bfloat16: "BF16",
        torch.int8: "I8",
        torch.int32: "I32",
        torch.int64: "I64",
    }

    dtype_size = {
        torch.float16: 2,
        torch.float32: 4,
        torch.bfloat16: 2,
        torch.int8: 1,
        torch.int32: 4,
        torch.int64: 8,
    }

    for name, tensor in tensors:
        nbytes = tensor.numel() * dtype_size[tensor.dtype]
        header[name] = {
            "dtype": dtype_map[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        tensor_data_parts.append(tensor.contiguous().cpu().numpy().tobytes())
        offset += nbytes

    header_json = json.dumps(header).encode("utf-8")
    # Pad header to 8-byte alignment (safetensors spec)
    while len(header_json) % 8 != 0:
        header_json += b" "

    header_size = len(header_json)
    tensor_data = b"".join(tensor_data_parts)

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", header_size))
        f.write(header_json)
        f.write(tensor_data)


def test_basic_correctness(sf_path, device_idx=0):
    """Test that io_uring O_DIRECT matches mmap for every tensor."""
    from safetensors import safe_open

    device = f"cuda:{device_idx}"
    errors = []

    # Load with mmap (ground truth - no alignment issues)
    mmap_tensors = {}
    with safe_open(str(sf_path), framework="pt", device=device, backend="mmap") as sf:
        for name in sf.keys():
            mmap_tensors[name] = sf.get_tensor(name)

    # Load with io_uring (O_DIRECT with alignment)
    uring_tensors = {}
    with safe_open(str(sf_path), framework="pt", device=device, backend="io_uring") as sf:
        for name in sf.keys():
            uring_tensors[name] = sf.get_tensor(name)

    # Compare every tensor
    assert set(mmap_tensors.keys()) == set(uring_tensors.keys()), "Key mismatch"

    for name in mmap_tensors:
        m = mmap_tensors[name]
        u = uring_tensors[name]

        if m.shape != u.shape:
            errors.append(f"{name}: shape mismatch {m.shape} vs {u.shape}")
            continue

        if m.dtype != u.dtype:
            errors.append(f"{name}: dtype mismatch {m.dtype} vs {u.dtype}")
            continue

        if not torch.equal(m, u):
            # Find where they differ
            diff_mask = m != u
            n_diff = diff_mask.sum().item()
            total = m.numel()
            errors.append(
                f"{name}: data mismatch ({n_diff}/{total} elements differ, "
                f"shape={m.shape}, dtype={m.dtype}, "
                f"nbytes={m.numel() * m.element_size()})"
            )

    return errors


def test_multi_device_correctness(sf_path, num_gpus):
    """Test that per-device routing via cudaSetDevice produces correct data."""
    from safetensors import safe_open

    errors = []

    # Get tensor names
    with safe_open(str(sf_path), framework="pt") as sf:
        all_names = list(sf.keys())

    if not all_names:
        return errors

    # Create device map: round-robin across GPUs
    device_map = {}
    for i, name in enumerate(all_names):
        device_map[name] = f"cuda:{i % num_gpus}"

    # Load with mmap (ground truth) — single device at a time
    mmap_tensors = {}
    for gpu_idx in range(num_gpus):
        dev = f"cuda:{gpu_idx}"
        names_for_dev = [n for n in all_names if device_map[n] == dev]
        with safe_open(str(sf_path), framework="pt", device=dev, backend="mmap") as sf:
            for name in names_for_dev:
                mmap_tensors[name] = sf.get_tensor(name)

    # Load with io_uring + device_map (tests cudaSetDevice per-tensor + O_DIRECT)
    uring_tensors = {}
    with safe_open(str(sf_path), framework="pt", device=device_map, backend="io_uring") as sf:
        for name in sf.keys():
            uring_tensors[name] = sf.get_tensor(name)

    # Compare
    for name in all_names:
        m = mmap_tensors[name]
        u = uring_tensors[name]

        if m.device != u.device:
            errors.append(f"{name}: device mismatch {m.device} vs {u.device}")
            continue

        expected_dev = device_map[name]
        if str(u.device) != expected_dev:
            errors.append(f"{name}: wrong device {u.device}, expected {expected_dev}")

        if not torch.equal(m, u):
            diff_mask = m != u
            n_diff = diff_mask.sum().item()
            errors.append(
                f"{name}: data mismatch on {u.device} ({n_diff}/{m.numel()} elements differ)"
            )

    return errors


def test_with_synthetic_tensors(tmpdir, device_idx=0):
    """Create synthetic tensors with specific sizes to test alignment edge cases.

    The safetensors header size affects where tensor data starts in the file.
    With different header sizes, tensors land at different file offsets, some
    of which are NOT aligned to 4096 bytes (the O_DIRECT requirement).
    """
    errors = []

    # Test cases: (tensor_name, shape, dtype)
    # These produce various tensor sizes to exercise alignment boundaries
    test_cases = [
        # Tiny tensors (< 4096 bytes) — entire tensor fits in alignment padding
        ("tiny_1", (1,), torch.float32),  # 4 bytes
        ("tiny_7", (7,), torch.float16),  # 14 bytes
        ("tiny_1023", (1023,), torch.int8),  # 1023 bytes — just under 1K
        ("tiny_4095", (4095,), torch.int8),  # 4095 bytes — just under page size

        # Page-boundary tensors
        ("page_exact", (1024,), torch.float32),  # 4096 bytes = 1 page
        ("page_plus_1", (2049,), torch.float16),  # 4098 bytes
        ("page_minus_1", (2047,), torch.float16),  # 4094 bytes

        # Multi-chunk tensors (> 8MB to span io_uring staging buffers)
        ("large_8mb", (2 * 1024 * 1024,), torch.float32),  # 8MB exact
        ("large_8mb_plus", (2 * 1024 * 1024 + 1,), torch.float32),  # 8MB + 4 bytes
        ("large_16mb", (4 * 1024 * 1024,), torch.float32),  # 16MB (2 chunks)
        ("large_24mb", (6 * 1024 * 1024,), torch.float32),  # 24MB (3 chunks)

        # Realistic model tensor sizes
        ("weight_5120x5120", (5120, 5120), torch.float16),  # 50MB — common in LLMs
        ("bias_5120", (5120,), torch.float16),  # 10KB
        ("embed_152064x5120", (152064, 5120), torch.float16),  # ~1.5GB-ish shape but smaller

        # Odd sizes that create interesting alignment patterns
        ("odd_prime", (7919,), torch.float32),  # prime number of elements
        ("odd_1byte", (8191,), torch.int8),  # prime number of bytes
    ]

    # Generate deterministic random data for each tensor
    torch.manual_seed(42)
    tensors = []
    for name, shape, dtype in test_cases:
        if dtype in (torch.float16, torch.float32, torch.bfloat16):
            t = torch.randn(shape, dtype=dtype)
        else:
            t = torch.randint(-128, 127, shape, dtype=dtype)
        tensors.append((name, t))

    # Write to safetensors file
    sf_path = os.path.join(tmpdir, "test_alignment.safetensors")
    create_test_safetensors(sf_path, tensors)

    # Verify file was written correctly
    total_size = os.path.getsize(sf_path)
    print(f"  Test file: {total_size / 1024 / 1024:.2f} MB, {len(tensors)} tensors")

    # Show tensor offsets in file for debugging
    from safetensors import safe_open
    with safe_open(sf_path, framework="pt") as sf:
        for name in sf.keys():
            info = sf.metadata()  # Not available in all versions
            pass

    # Run the comparison
    basic_errors = test_basic_correctness(sf_path, device_idx)
    errors.extend(basic_errors)

    return errors


def test_with_real_model(model_path, num_gpus):
    """Test with actual model files (most realistic test)."""
    sf_files = sorted(Path(model_path).glob("*.safetensors"))
    if not sf_files:
        print("  No safetensors files found, skipping")
        return []

    errors = []
    for sf_file in sf_files:
        size_mb = os.path.getsize(sf_file) / 1024 / 1024
        print(f"  Testing {sf_file.name} ({size_mb:.1f} MB)...")

        # Single device test
        file_errors = test_basic_correctness(sf_file, device_idx=0)
        if file_errors:
            errors.extend([f"[{sf_file.name}] {e}" for e in file_errors])

        # Multi-device test (if multiple GPUs)
        if num_gpus > 1:
            md_errors = test_multi_device_correctness(sf_file, num_gpus)
            if md_errors:
                errors.extend([f"[{sf_file.name} multi-dev] {e}" for e in md_errors])

    return errors


def main():
    parser = argparse.ArgumentParser(description="O_DIRECT Data Correctness Test")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model directory with .safetensors files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model ID to test with",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Skip synthetic tensor tests",
    )
    parser.add_argument(
        "--skip-real-model",
        action="store_true",
        help="Skip real model tests",
    )
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    print(f"GPUs available: {num_gpus}")
    print()

    all_errors = []

    # Test 1: Synthetic tensors with specific alignment patterns
    if not args.skip_synthetic:
        print("=" * 60)
        print("Test 1: Synthetic tensors (alignment edge cases)")
        print("=" * 60)
        with tempfile.TemporaryDirectory() as tmpdir:
            errors = test_with_synthetic_tensors(tmpdir, device_idx=0)
            if errors:
                print(f"  FAILED: {len(errors)} errors")
                for e in errors:
                    print(f"    - {e}")
                all_errors.extend(errors)
            else:
                print("  PASSED")

        # Test 1b: Multi-device with synthetic tensors
        if num_gpus > 1:
            print()
            print("Test 1b: Synthetic tensors (multi-device)")
            with tempfile.TemporaryDirectory() as tmpdir:
                torch.manual_seed(42)
                tensors = [
                    (f"layer_{i}_weight", torch.randn(1024, 1024, dtype=torch.float16))
                    for i in range(num_gpus * 4)
                ]
                sf_path = os.path.join(tmpdir, "test_multidev.safetensors")
                create_test_safetensors(sf_path, tensors)
                errors = test_multi_device_correctness(sf_path, num_gpus)
                if errors:
                    print(f"  FAILED: {len(errors)} errors")
                    for e in errors:
                        print(f"    - {e}")
                    all_errors.extend(errors)
                else:
                    print(f"  PASSED ({len(tensors)} tensors across {num_gpus} GPUs)")
        print()

    # Test 2: Real model files
    if not args.skip_real_model:
        model_path = args.model_path
        if not model_path and args.model:
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(
                args.model,
                cache_dir=args.cache_dir,
                allow_patterns=["*.safetensors", "*.json"],
                ignore_patterns=["*.bin", "*.pt", "*.onnx"],
            )

        if model_path:
            print("=" * 60)
            print(f"Test 2: Real model ({model_path})")
            print("=" * 60)
            errors = test_with_real_model(model_path, num_gpus)
            if errors:
                print(f"  FAILED: {len(errors)} errors")
                for e in errors[:20]:
                    print(f"    - {e}")
                if len(errors) > 20:
                    print(f"    ... and {len(errors) - 20} more")
                all_errors.extend(errors)
            else:
                print("  ALL PASSED")
            print()

    # Summary
    print("=" * 60)
    if all_errors:
        print(f"FAILED: {len(all_errors)} total errors")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()

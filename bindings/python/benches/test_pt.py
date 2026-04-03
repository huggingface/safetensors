import os
import sys
import tempfile

import pytest
import torch

from safetensors.torch import load_file, load_file_io_uring, save_file


def create_gpt2(n_layers: int):
    tensors = {}
    tensors["wte"] = torch.zeros((50257, 768))
    tensors["wpe"] = torch.zeros((1024, 768))
    for i in range(n_layers):
        tensors[f"h.{i}.ln_1.weight"] = torch.zeros((768,))
        tensors[f"h.{i}.ln_1.bias"] = torch.zeros((768,))
        tensors[f"h.{i}.attn.bias"] = torch.zeros((1, 1, 1024, 1024))
        tensors[f"h.{i}.attn.c_attn.weight"] = torch.zeros((768, 2304))
        tensors[f"h.{i}.attn.c_attn.bias"] = torch.zeros((2304))
        tensors[f"h.{i}.attn.c_proj.weight"] = torch.zeros((768, 768))
        tensors[f"h.{i}.attn.c_proj.bias"] = torch.zeros((768))
        tensors[f"h.{i}.ln_2.weight"] = torch.zeros((768))
        tensors[f"h.{i}.ln_2.bias"] = torch.zeros((768))
        tensors[f"h.{i}.mlp.c_fc.weight"] = torch.zeros((768, 3072))
        tensors[f"h.{i}.mlp.c_fc.bias"] = torch.zeros((3072))
        tensors[f"h.{i}.mlp.c_proj.weight"] = torch.zeros((3072, 768))
        tensors[f"h.{i}.mlp.c_proj.bias"] = torch.zeros((768))
    tensors["ln_f.weight"] = torch.zeros((768))
    tensors["ln_f.bias"] = torch.zeros((768))
    return tensors


def create_lora(n_layers: int):
    tensors = {}
    for i in range(n_layers):
        tensors[f"lora.{i}.up.weight"] = torch.zeros((32, 32))
        tensors[f"lora.{i}.down.weight"] = torch.zeros((32, 32))
    return tensors


def test_pt_pt_load_cpu(benchmark):
    # benchmark something
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        torch.save(weights, f)
        result = benchmark(torch.load, f.name)
    os.unlink(f.name)

    for k, v in weights.items():
        tv = result[k]
        assert torch.allclose(v, tv)


def test_pt_sf_load_cpu(benchmark):
    # benchmark something
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)
        result = benchmark(load_file, f.name)
    os.unlink(f.name)

    for k, v in weights.items():
        tv = result[k]
        assert torch.allclose(v, tv)


def test_pt_pt_load_cpu_small(benchmark):
    weights = create_lora(500)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        torch.save(weights, f)
        result = benchmark(torch.load, f.name)
    os.unlink(f.name)

    for k, v in weights.items():
        tv = result[k]
        assert torch.allclose(v, tv)


def test_pt_sf_load_cpu_small(benchmark):
    weights = create_lora(500)

    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)
        result = benchmark(load_file, f.name)
    os.unlink(f.name)

    for k, v in weights.items():
        tv = result[k]
        assert torch.allclose(v, tv)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_pt_pt_load_gpu(benchmark):
    # benchmark something
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        torch.save(weights, f)
        result = benchmark(torch.load, f.name, map_location="cuda:0")
    os.unlink(f.name)

    for k, v in weights.items():
        v = v.cuda()
        tv = result[k]
        assert torch.allclose(v, tv)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_pt_sf_load_gpu(benchmark):
    # benchmark something
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)
        result = benchmark(load_file, f.name, device="cuda:0")
    os.unlink(f.name)

    for k, v in weights.items():
        v = v.cuda()
        tv = result[k]
        assert torch.allclose(v, tv)


@pytest.mark.skipif(
    not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available(),
    reason="requires mps",
)
def test_pt_pt_load_mps(benchmark):
    # benchmark something
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        torch.save(weights, f)
        result = benchmark(torch.load, f.name, map_location="mps")
    os.unlink(f.name)

    for k, v in weights.items():
        v = v.to(device="mps")
        tv = result[k]
        assert torch.allclose(v, tv)


@pytest.mark.skipif(
    not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available(),
    reason="requires mps",
)
def test_pt_sf_load_mps(benchmark):
    # benchmark something
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)
        result = benchmark(load_file, f.name, device="mps")
    os.unlink(f.name)

    for k, v in weights.items():
        v = v.to(device="mps")
        tv = result[k]
        assert torch.allclose(v, tv)


def test_pt_sf_save_cpu(benchmark):
    weights = create_gpt2(12)

    filename = "tmp.safetensors"

    # XXX: On some platforms (tested on Linux x86_64 ext4), writing to an already existing file is slower than creating a new one.
    # On others, such as MacOS (APFS), it's the opposite. To have more consistent benchmarks,
    # we ensure the file does not exist before each write, which is also closer to real world usage.
    def setup():
        try:
            os.unlink(filename)
        except Exception:
            pass

    benchmark.pedantic(
        save_file, args=(weights, filename), setup=setup, iterations=1, rounds=5
    )

    # Clean up files
    os.unlink(filename)


def test_pt_sf_load_cpu_linux_io_uring(benchmark):
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)
        result = benchmark(load_file_io_uring, f.name)
    os.unlink(f.name)

    for k, v in weights.items():
        tv = result[k]
        assert torch.allclose(v, tv)


# ============================================================================
# Scaling Benchmarks - Test mmap vs io_uring across different file sizes
# ============================================================================


def create_large_model(size_mb: int):
    """Create a model of approximately size_mb megabytes."""
    # F32 = 4 bytes per element
    # Target approximately size_mb worth of tensors
    target_bytes = size_mb * 1024 * 1024
    elements_per_tensor = target_bytes // 4  # F32 = 4 bytes
    dim = int(elements_per_tensor**0.5)

    tensors = {}
    tensors["weight"] = torch.randn((dim, dim))
    return tensors


# 500MB Benchmarks
@pytest.fixture(scope="module")
def file_500mb():
    """Create a 500MB test file."""
    weights = create_large_model(500)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as f:
        save_file(weights, f.name)
        yield f.name
    os.unlink(f.name)


def load_and_access_all_data(load_fn, filepath):
    """Load file and access ALL data to force mmap to actually read from disk."""
    result = load_fn(filepath)
    # Force reading all data by computing sum of all tensor elements
    total = 0.0
    for tensor in result.values():
        total += tensor.sum().item()
    return result, total


def test_pt_sf_load_500mb_mmap(benchmark, file_500mb):
    """Benchmark mmap loading of 500MB file (with full data access)."""
    result, total = benchmark(load_and_access_all_data, load_file, file_500mb)
    assert len(result) > 0


def test_pt_sf_load_500mb_io_uring(benchmark, file_500mb):
    """Benchmark io_uring loading of 500MB file (with full data access)."""
    if sys.platform != "linux":
        pytest.skip("io_uring only available on Linux")
    result, total = benchmark(load_and_access_all_data, load_file_io_uring, file_500mb)
    assert len(result) > 0


# 1GB Benchmarks
@pytest.fixture(scope="module")
def file_1gb():
    """Create a 1GB test file."""
    weights = create_large_model(1000)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as f:
        save_file(weights, f.name)
        yield f.name
    os.unlink(f.name)


def test_pt_sf_load_1gb_mmap(benchmark, file_1gb):
    """Benchmark mmap loading of 1GB file (with full data access)."""
    # Use fewer iterations for large files
    benchmark.pedantic(
        load_and_access_all_data, args=(load_file, file_1gb), iterations=3, rounds=1
    )


def test_pt_sf_load_1gb_io_uring(benchmark, file_1gb):
    """Benchmark io_uring loading of 1GB file (with full data access)."""
    if sys.platform != "linux":
        pytest.skip("io_uring only available on Linux")
    # Use fewer iterations for large files
    benchmark.pedantic(
        load_and_access_all_data,
        args=(load_file_io_uring, file_1gb),
        iterations=3,
        rounds=1,
    )

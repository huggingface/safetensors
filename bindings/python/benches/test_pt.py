import os
import tempfile

import pytest
import torch

from safetensors.torch import load_file, save_file

import subprocess


def drop_linux_page_cache():
    """
    Properly drop Linux page cache using sudo and tee.
    """

    print("dropping os page cache")

    subprocess.run(["sync"], check=True)

    proc = subprocess.run(
        ["sudo", "tee", "/proc/sys/vm/drop_caches"],
        input="3\n",
        text=True,
        capture_output=True,
    )

    print(f"stdout: {proc.stdout}")
    print(f"stderr: {proc.stderr}")

    if proc.returncode != 0:
        raise RuntimeError(f"Failed to drop caches: {proc.stderr.strip()}")


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
        result = benchmark(load_file, f.name, direct=False)
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
        result = benchmark(load_file, f.name, direct=False)
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
        result = benchmark(load_file, f.name, device="cuda:0", direct=False)
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
        result = benchmark(load_file, f.name, device="mps", direct=False)
    os.unlink(f.name)

    for k, v in weights.items():
        v = v.to(device="mps")
        tv = result[k]
        assert torch.allclose(v, tv)


def test_pt_sf_load_direct(benchmark):
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)
        result = benchmark(load_file, f.name, direct=True)
    os.unlink(f.name)

    for k, v in weights.items():
        tv = result[k]
        assert torch.allclose(v, tv)


# TODO: cleanup file after benchmark
def test_pt_mmap_no_warmup(benchmark):
    def setup():
        with tempfile.NamedTemporaryFile(delete=False) as f:
            save_file(create_gpt2(12), f.name)
            drop_linux_page_cache()
        return (f.name,), {"direct": False}

    benchmark.pedantic(load_file, setup=setup, rounds=1, warmup_rounds=0)


def load_and_touch(path, direct):
    tensors = load_file(path, direct)
    acc = 0

    for t in tensors.values():
        acc += t.sum().item()

    return path


def test_pt_sf_load_and_touch_direct(benchmark):
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)
        result = benchmark(load_and_touch, f.name, direct=True)

    os.unlink(f.name)

    for k, v in weights.items():
        tv = result[k]
        assert torch.allclose(v, tv)


def test_pt_sf_load_and_touch_mmap(benchmark):
    def setup():
        with tempfile.NamedTemporaryFile(delete=False) as f:
            save_file(create_gpt2(12), f.name)
            drop_linux_page_cache()
        return (f.name,), {"direct": False}

    path = benchmark.pedantic(load_and_touch, setup=setup, rounds=1, warmup_rounds=0)

    os.unlink(path)

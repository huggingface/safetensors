import os
import tempfile

import pytest
import torch

from safetensors.torch import load_file, save_file


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


def create_llama(
    n_layers: int,
    hidden_size: int = 4096,
    intermediate_size: int = 11008,
    vocab_size: int = 32000,
):
    """Create Llama-style model weights (7B-ish with 32 layers by default)."""
    tensors = {}
    tensors["model.embed_tokens.weight"] = torch.zeros((vocab_size, hidden_size))
    for i in range(n_layers):
        tensors[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.zeros(
            (hidden_size, hidden_size)
        )
        tensors[f"model.layers.{i}.self_attn.k_proj.weight"] = torch.zeros(
            (hidden_size, hidden_size)
        )
        tensors[f"model.layers.{i}.self_attn.v_proj.weight"] = torch.zeros(
            (hidden_size, hidden_size)
        )
        tensors[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.zeros(
            (hidden_size, hidden_size)
        )
        tensors[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.zeros(
            (intermediate_size, hidden_size)
        )
        tensors[f"model.layers.{i}.mlp.up_proj.weight"] = torch.zeros(
            (intermediate_size, hidden_size)
        )
        tensors[f"model.layers.{i}.mlp.down_proj.weight"] = torch.zeros(
            (hidden_size, intermediate_size)
        )
        tensors[f"model.layers.{i}.input_layernorm.weight"] = torch.zeros(
            (hidden_size,)
        )
        tensors[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.zeros(
            (hidden_size,)
        )
    tensors["model.norm.weight"] = torch.zeros((hidden_size,))
    tensors["lm_head.weight"] = torch.zeros((vocab_size, hidden_size))
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


def test_pt_sf_load_cpu_with_sum(benchmark):
    """Benchmark CPU loading with actual data access via .sum().

    This ensures we measure real memory access, not just mmap setup.
    """
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)

        def load_and_sum():
            result = load_file(f.name)
            total = sum(t.sum().item() for t in result.values())
            return result, total

        result, _ = benchmark(load_and_sum)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_pt_sf_load_gpu_io_uring(benchmark):
    """Benchmark GPU loading with io_uring backend (Linux only).

    Note: io_uring uses staging buffers + async memcpy, which has more overhead
    than mmap's direct cudaMemcpy for sequential loading. io_uring is optimized
    for high-concurrency async I/O patterns, not single-file sequential reads.
    """
    from safetensors import safe_open

    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)

        def load_with_io_uring():
            tensors = {}
            with safe_open(
                f.name, framework="pt", device="cuda:0", backend="io_uring"
            ) as sf:
                for key in sf.keys():
                    tensors[key] = sf.get_tensor(key)
            return tensors

        result = benchmark(load_with_io_uring)
    os.unlink(f.name)

    for k, v in weights.items():
        v = v.cuda()
        tv = result[k]
        assert torch.allclose(v, tv)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_pt_sf_load_gpu_llama(benchmark):
    """Benchmark GPU loading with Llama-sized model (~16GB).

    Tests loading performance for larger models.
    """
    # Use 20 layers (~16GB) to fit in 23GB GPU with headroom
    weights = create_llama(20)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)
        result = benchmark(load_file, f.name, device="cuda:0")
    os.unlink(f.name)

    for k, v in weights.items():
        v = v.cuda()
        tv = result[k]
        assert torch.allclose(v, tv)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_pt_sf_load_gpu_llama_io_uring(benchmark):
    """Benchmark GPU loading with Llama-sized model using io_uring."""
    from safetensors import safe_open

    weights = create_llama(20)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)

        def load_with_io_uring():
            tensors = {}
            with safe_open(
                f.name, framework="pt", device="cuda:0", backend="io_uring"
            ) as sf:
                for key in sf.keys():
                    tensors[key] = sf.get_tensor(key)
            return tensors

        result = benchmark(load_with_io_uring)
    os.unlink(f.name)

    for k, v in weights.items():
        v = v.cuda()
        tv = result[k]
        assert torch.allclose(v, tv)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_pt_sf_load_gpu_iter_tensors(benchmark):
    """Benchmark GPU loading with iter_tensors (async prefetch).

    Comparable to test_pt_sf_load_gpu - uses GPT2-12 model.
    """
    from safetensors import safe_open

    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)

        def load_with_iter_tensors():
            tensors = {}
            with safe_open(f.name, framework="pt", device="cuda:0") as sf:
                for name, tensor in sf.iter_tensors(prefetch=4):
                    tensors[name] = tensor
            return tensors

        result = benchmark(load_with_iter_tensors)
    os.unlink(f.name)

    for k, v in weights.items():
        v = v.cuda()
        tv = result[k]
        assert torch.allclose(v, tv)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_pt_sf_load_gpu_get_tensor(benchmark):
    """Benchmark GPU loading with get_tensor loop (no prefetch).

    Comparable to test_pt_sf_load_gpu - uses GPT2-12 model.
    """
    from safetensors import safe_open

    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)

        def load_with_get_tensor():
            tensors = {}
            with safe_open(f.name, framework="pt", device="cuda:0") as sf:
                for key in sf.keys():
                    tensors[key] = sf.get_tensor(key)
            return tensors

        result = benchmark(load_with_get_tensor)
    os.unlink(f.name)

    for k, v in weights.items():
        v = v.cuda()
        tv = result[k]
        assert torch.allclose(v, tv)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_pt_sf_load_gpu_llama_iter_tensors(benchmark):
    """Benchmark GPU loading with Llama-sized model using iter_tensors.

    Comparable to test_pt_sf_load_gpu_llama.
    """
    from safetensors import safe_open

    weights = create_llama(20)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)

        def load_with_iter_tensors():
            tensors = {}
            with safe_open(f.name, framework="pt", device="cuda:0") as sf:
                for name, tensor in sf.iter_tensors(prefetch=4):
                    tensors[name] = tensor
            return tensors

        result = benchmark(load_with_iter_tensors)
    os.unlink(f.name)

    for k, v in weights.items():
        v = v.cuda()
        tv = result[k]
        assert torch.allclose(v, tv)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_pt_sf_load_gpu_llama_get_tensor(benchmark):
    """Benchmark GPU loading with Llama-sized model using get_tensor loop.

    Comparable to test_pt_sf_load_gpu_llama.
    """
    from safetensors import safe_open

    weights = create_llama(20)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)

        def load_with_get_tensor():
            tensors = {}
            with safe_open(f.name, framework="pt", device="cuda:0") as sf:
                for key in sf.keys():
                    tensors[key] = sf.get_tensor(key)
            return tensors

        result = benchmark(load_with_get_tensor)
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


def test_pt_sf_iter_tensors_cpu(benchmark):
    """Benchmark iter_tensors vs get_tensor loop on CPU."""
    from safetensors import safe_open

    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)

        def load_with_iter_tensors():
            tensors = {}
            with safe_open(f.name, framework="pt", device="cpu") as sf:
                for name, tensor in sf.iter_tensors(prefetch=4):
                    tensors[name] = tensor
            return tensors

        result = benchmark(load_with_iter_tensors)
    os.unlink(f.name)

    for k, v in weights.items():
        tv = result[k]
        assert torch.allclose(v, tv)


def test_pt_sf_iter_tensors_cpu_with_work(benchmark):
    """Benchmark iter_tensors with simulated processing (e.g., quantization).

    This measures the benefit of prefetching when there's actual work between loads.
    """
    from safetensors import safe_open

    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)

        def load_with_work():
            tensors = {}
            with safe_open(f.name, framework="pt", device="cpu") as sf:
                for name, tensor in sf.iter_tensors(prefetch=4):
                    # Simulate quantization-like work
                    tensor = tensor.half().float()
                    tensors[name] = tensor
            return tensors

        result = benchmark(load_with_work)
    os.unlink(f.name)

    for k in weights.keys():
        assert k in result


def test_pt_sf_get_tensor_cpu_with_work(benchmark):
    """Benchmark traditional get_tensor loop with simulated processing.

    Compare against iter_tensors_cpu_with_work to see prefetch benefit.
    """
    from safetensors import safe_open

    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)

        def load_with_work():
            tensors = {}
            with safe_open(f.name, framework="pt", device="cpu") as sf:
                for name in sf.keys():
                    tensor = sf.get_tensor(name)
                    # Simulate quantization-like work
                    tensor = tensor.half().float()
                    tensors[name] = tensor
            return tensors

        result = benchmark(load_with_work)
    os.unlink(f.name)

    for k in weights.keys():
        assert k in result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_pt_sf_iter_tensors_gpu(benchmark):
    """Benchmark iter_tensors on GPU."""
    from safetensors import safe_open

    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)

        def load_with_iter_tensors():
            tensors = {}
            with safe_open(f.name, framework="pt", device="cuda:0") as sf:
                for name, tensor in sf.iter_tensors(prefetch=4):
                    tensors[name] = tensor
            return tensors

        result = benchmark(load_with_iter_tensors)
    os.unlink(f.name)

    for k, v in weights.items():
        v = v.cuda()
        tv = result[k]
        assert torch.allclose(v, tv)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_pt_sf_iter_tensors_gpu_with_work(benchmark):
    """Benchmark iter_tensors on GPU with simulated quantization.

    This is the key benchmark: shows prefetch benefit when GPU compute
    overlaps with next tensor's I/O + allocation.
    """
    from safetensors import safe_open

    weights = create_llama(32)  # Larger model for more realistic test
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)

        def load_with_work():
            tensors = {}
            with safe_open(f.name, framework="pt", device="cuda:0") as sf:
                for name, tensor in sf.iter_tensors(prefetch=4):
                    # Simulate quantization
                    tensor = tensor.half()
                    tensors[name] = tensor
            torch.cuda.synchronize()
            return tensors

        result = benchmark(load_with_work)
    os.unlink(f.name)

    for k in weights.keys():
        assert k in result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_pt_sf_get_tensor_gpu_with_work(benchmark):
    """Benchmark traditional get_tensor on GPU with simulated quantization.

    Compare against iter_tensors_gpu_with_work.
    """
    from safetensors import safe_open

    weights = create_llama(32)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        save_file(weights, f.name)

        def load_with_work():
            tensors = {}
            with safe_open(f.name, framework="pt", device="cuda:0") as sf:
                for name in sf.keys():
                    tensor = sf.get_tensor(name)
                    # Simulate quantization
                    tensor = tensor.half()
                    tensors[name] = tensor
            torch.cuda.synchronize()
            return tensors

        result = benchmark(load_with_work)
    os.unlink(f.name)

    for k in weights.keys():
        assert k in result

import tempfile

import jax.numpy as jnp
from flax.serialization import msgpack_restore, msgpack_serialize
from safetensors.flax import load_file, save_file


def create_gpt2(n_layers: int):
    tensors = {}
    tensors["wte"] = jnp.zeros((50257, 768))
    tensors["wpe"] = jnp.zeros((1024, 768))
    for i in range(n_layers):
        tensors[f"h.{i}.ln_1.weight"] = jnp.zeros((768,))
        tensors[f"h.{i}.ln_1.bias"] = jnp.zeros((768,))
        tensors[f"h.{i}.attn.bias"] = jnp.zeros((1, 1, 1024, 1024))
        tensors[f"h.{i}.attn.c_attn.weight"] = jnp.zeros((768, 2304))
        tensors[f"h.{i}.attn.c_attn.bias"] = jnp.zeros((2304))
        tensors[f"h.{i}.attn.c_proj.weight"] = jnp.zeros((768, 768))
        tensors[f"h.{i}.attn.c_proj.bias"] = jnp.zeros((768))
        tensors[f"h.{i}.ln_2.weight"] = jnp.zeros((768))
        tensors[f"h.{i}.ln_2.bias"] = jnp.zeros((768))
        tensors[f"h.{i}.mlp.c_fc.weight"] = jnp.zeros((768, 3072))
        tensors[f"h.{i}.mlp.c_fc.bias"] = jnp.zeros((3072))
        tensors[f"h.{i}.mlp.c_proj.weight"] = jnp.zeros((3072, 768))
        tensors[f"h.{i}.mlp.c_proj.bias"] = jnp.zeros((768))
    tensors["ln_f.weight"] = jnp.zeros((768))
    tensors["ln_f.bias"] = jnp.zeros((768))
    return tensors


def load(filename):
    with open(filename, "rb") as f:
        data = f.read()
        flax_weights = msgpack_restore(data)
        return flax_weights


def test_flax_flax_load(benchmark):
    # benchmark something
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile() as f:
        serialized = msgpack_serialize(weights)
        f.write(serialized)
        result = benchmark(load, f.name)

    for k, v in weights.items():
        tv = result[k]
        assert jnp.allclose(v, tv)


def test_flax_sf_load(benchmark):
    # benchmark something
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile() as f:
        save_file(weights, f.name)
        result = benchmark(load_file, f.name)

    for k, v in weights.items():
        tv = result[k]
        assert jnp.allclose(v, tv)

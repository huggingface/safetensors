import tempfile

import h5py
import numpy as np
import tensorflow as tf

from safetensors.tensorflow import load_file, save_file


def _load(filename, tensors=None, prefix=""):
    with h5py.File(filename, "r") as f:
        if tensors is None:
            tensors = {}
        for k in f.keys():
            if isinstance(f[k], h5py._hl.dataset.Dataset):
                key = k if not prefix else f"{prefix}_{k}"
                tensors[key] = tf.convert_to_tensor(np.array(f[k]))
            else:
                tensors.update(_load(f[k], tensors, prefix=f"{prefix}_{k}"))
        return tensors


def _save(filename, tensors, prefix=""):
    with h5py.File(filename, "w") as f:
        for name, tensor in tensors.items():
            tensor = tensor.numpy()
            dset = f.create_dataset(name, tensor.shape, dtype=tensor.dtype)
            dset[:] = tensor


def create_gpt2(n_layers: int):
    tensors = {}
    tensors["wte"] = tf.zeros((50257, 768))
    tensors["wpe"] = tf.zeros((1024, 768))
    for i in range(n_layers):
        tensors[f"h.{i}.ln_1.weight"] = tf.zeros((768,))
        tensors[f"h.{i}.ln_1.bias"] = tf.zeros((768,))
        tensors[f"h.{i}.attn.bias"] = tf.zeros((1, 1, 1024, 1024))
        tensors[f"h.{i}.attn.c_attn.weight"] = tf.zeros((768, 2304))
        tensors[f"h.{i}.attn.c_attn.bias"] = tf.zeros((2304))
        tensors[f"h.{i}.attn.c_proj.weight"] = tf.zeros((768, 768))
        tensors[f"h.{i}.attn.c_proj.bias"] = tf.zeros((768))
        tensors[f"h.{i}.ln_2.weight"] = tf.zeros((768))
        tensors[f"h.{i}.ln_2.bias"] = tf.zeros((768))
        tensors[f"h.{i}.mlp.c_fc.weight"] = tf.zeros((768, 3072))
        tensors[f"h.{i}.mlp.c_fc.bias"] = tf.zeros((3072))
        tensors[f"h.{i}.mlp.c_proj.weight"] = tf.zeros((3072, 768))
        tensors[f"h.{i}.mlp.c_proj.bias"] = tf.zeros((768))
    tensors["ln_f.weight"] = tf.zeros((768))
    tensors["ln_f.bias"] = tf.zeros((768))
    return tensors


def test_tf_tf_load(benchmark):
    # benchmark something
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile() as f:
        _save(f.name, weights)
        result = benchmark(_load, f.name)

    for k, v in weights.items():
        tv = result[k]
        assert np.allclose(v, tv)


def test_tf_sf_load(benchmark):
    # benchmark something
    weights = create_gpt2(12)
    with tempfile.NamedTemporaryFile() as f:
        save_file(weights, f.name)
        result = benchmark(load_file, f.name, device="cpu")

    for k, v in weights.items():
        tv = result[k]
        assert np.allclose(v, tv)

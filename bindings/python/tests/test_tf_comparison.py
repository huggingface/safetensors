import unittest

import h5py
import numpy as np
import tensorflow as tf

from safetensors import safe_open
from safetensors.tensorflow import load_file, save_file


def _load(f, tensors=None, prefix=""):
    if tensors is None:
        tensors = {}
    for k in f.keys():
        if isinstance(f[k], h5py._hl.dataset.Dataset):
            key = k if not prefix else f"{prefix}_{k}"
            tensors[key] = tf.convert_to_tensor(np.array(f[k]))
        else:
            tensors.update(_load(f[k], tensors, prefix=f"{prefix}_{k}"))
    return tensors


def _save(f, tensors, prefix=""):
    for name, tensor in tensors.items():
        tensor = tensor.numpy()
        dset = f.create_dataset(name, tensor.shape, dtype=tensor.dtype)
        dset[:] = tensor


class SafeTestCase(unittest.TestCase):
    def setUp(self):
        data = {
            "test": tf.zeros((1024, 1024), dtype=tf.float32),
            "test2": tf.zeros((1024, 1024), dtype=tf.float32),
            "test3": tf.zeros((1024, 1024), dtype=tf.float32),
            "test4": tf.zeros((1024, 1024), dtype=tf.complex64),
        }
        self.tf_filename = "./tests/data/tf_load.h5"
        self.sf_filename = "./tests/data/tf_load.safetensors"

        with h5py.File(self.tf_filename, "w") as f:
            _save(f, data)
        save_file(data, self.sf_filename)

    def test_zero_sized(self):
        data = {
            "test": tf.zeros((2, 0), dtype=tf.float32),
        }
        local = "./tests/data/out_safe_flat_mmap_small2.safetensors"
        save_file(data.copy(), local)
        reloaded = load_file(local)
        # Empty tensor != empty tensor on numpy, so comparing shapes
        # instead
        self.assertEqual(data["test"].shape, reloaded["test"].shape)

    def test_deserialization_safe(self):
        weights = load_file(self.sf_filename)

        with h5py.File(self.tf_filename, "r") as f:
            tf_weights = _load(f)

        for k, v in weights.items():
            tv = tf_weights[k]
            self.assertTrue(np.allclose(v, tv))

    def test_bfloat16(self):
        data = {
            "test": tf.random.normal((1024, 1024), dtype=tf.bfloat16),
        }
        save_file(data, self.sf_filename)
        weights = {}
        with safe_open(self.sf_filename, framework="tf") as f:
            for k in f.keys():
                weights[k] = f.get_tensor(k)

        for k, v in weights.items():
            tv = data[k]
            self.assertTrue(tf.experimental.numpy.allclose(v, tv))

    def test_deserialization_safe_open(self):
        weights = {}
        with safe_open(self.sf_filename, framework="tf") as f:
            for k in f.keys():
                weights[k] = f.get_tensor(k)

        with h5py.File(self.tf_filename, "r") as f:
            tf_weights = _load(f)

        for k, v in weights.items():
            tv = tf_weights[k]
            self.assertTrue(np.allclose(v, tv))

import datetime
import os
import unittest

import h5py
import numpy as np
import tensorflow as tf

from huggingface_hub import hf_hub_download
from safetensors.tensorflow import load_file, save_file


MODEL_ID = os.getenv("MODEL_ID", "gpt2")


def _load(f, tensors, prefix=""):
    for k in f.keys():
        if isinstance(f[k], h5py._hl.dataset.Dataset):
            tensors[f"{prefix}_{k}"] = tf.convert_to_tensor(np.array(f[k]))
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
        self.filename = hf_hub_download(MODEL_ID, filename="tf_model.h5")
        tensors = {}
        with h5py.File(self.filename, "r") as f:
            self.data = _load(f, tensors)

        # Weird copies are necessary because apparently calling
        # tf.Tensor(..).numpy() converts inplace destroying the actual
        # tf.Tensor.
        data = self.data.copy()
        self.local = "./tests/data/out_safe_tf_mmap.h5"
        save_file(data, self.local)

    def test_deserialization_safe(self):
        load_file(self.local)

        start = datetime.datetime.now()
        load_file(self.local)
        safe_time = datetime.datetime.now() - start

        tensors = {}
        with h5py.File(self.filename, "r") as f:
            self.data = _load(f, tensors)

        start = datetime.datetime.now()
        tensors = {}
        with h5py.File(self.filename, "r") as f:
            self.data = _load(f, tensors)
        tf_time = datetime.datetime.now() - start
        print()
        print(f"Deserialization (Safe) took {safe_time}")
        print(f"Deserialization (TF) took {tf_time} (Safe is {tf_time/safe_time} faster)")

    def test_serialization_safe(self):
        outfilename = "./tests/data/out_safe.safetensors"
        data = self.data.copy()
        save_file(data, outfilename)

        data = self.data.copy()
        start = datetime.datetime.now()
        save_file(data, outfilename)
        safe_time = datetime.datetime.now() - start

        data = self.data.copy()
        start = datetime.datetime.now()
        with h5py.File("./tests/data/out_tf_tf.h5", "w") as f:
            _save(f, data)
        tf_time = datetime.datetime.now() - start

        print()
        print(f"Serialization (Safe) took {safe_time}")
        print(f"Serialization (TF) took {tf_time} (Safe is {tf_time/safe_time} faster)")

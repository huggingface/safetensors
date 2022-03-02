import unittest
from safetensors.flax import save_file, load_file
from huggingface_hub import hf_hub_download
import datetime
import h5py
import numpy as np
import os

MODEL_ID = os.getenv("MODEL_ID", "gpt2")


def _load(f, tensors, prefix=""):
    for k in f.keys():
        if isinstance(f[k], h5py._hl.dataset.Dataset):
            tensors[f"{prefix}_{k}"] = np.array(f[k])
        else:
            tensors.update(_load(f[k], tensors, prefix=f"{prefix}_{k}"))
    return tensors


def _save(f, tensors, prefix=""):
    for name, tensor in tensors.items():
        dset = f.create_dataset(name, tensor.shape, dtype=tensor.dtype)
        dset[:] = tensor


class SafeTestCase(unittest.TestCase):
    def setUp(self):
        self.filename = hf_hub_download(MODEL_ID, filename="tf_model.h5")
        tensors = {}
        with h5py.File(self.filename, "r") as f:
            self.data = _load(f, tensors)

        print("TF keys", len(self.data.keys()))
        self.local = "./tests/data/out_safe_tf_mmap.h5"
        save_file(self.data, self.local)

    def test_deserialization_safe(self):
        start = datetime.datetime.now()
        load_file(self.local)
        print()
        print("Deserialization (Safe) took ", datetime.datetime.now() - start)

    def test_serialization_safe(self):
        start = datetime.datetime.now()
        outfilename = "./tests/data/out_safe.bin"
        save_file(self.data, outfilename)
        print()
        print("Serialization (Safe) took ", datetime.datetime.now() - start)


class TFTestCase(unittest.TestCase):
    def setUp(self):
        self.filename = hf_hub_download(MODEL_ID, filename="tf_model.h5")
        tensors = {}
        with h5py.File(self.filename, "r") as f:
            self.data = _load(f, tensors)
        self.local = "./tests/data/out_safe_tf_mmap.bin"
        save_file(self.data, self.local)

    def test_deserialization_tf(self):
        start = datetime.datetime.now()
        tensors = {}
        with h5py.File(self.filename, "r") as f:
            self.data = _load(f, tensors)
        print()
        print("Deserialization (TF) took ", datetime.datetime.now() - start)

    def test_serialization_tf(self):
        start = datetime.datetime.now()
        with h5py.File("./tests/data/out_tf_tf.h5", "w") as f:
            _save(f, self.data)
        print()
        print("Serialization (TF) took ", datetime.datetime.now() - start)

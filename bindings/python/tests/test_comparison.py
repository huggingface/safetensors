import unittest
from safetensors.torch import save_file, load_file, load
from huggingface_hub import hf_hub_download
import torch
import datetime
import os


MODEL_ID = os.getenv("MODEL_ID", "gpt2")


class BigTestCase(unittest.TestCase):
    def test_gpt2_deserialization_safe(self):
        filename = hf_hub_download(MODEL_ID, filename="pytorch_model.bin")
        data = torch.load(filename)

        local = "out_safe.bin"
        save_file(data, local)

        start = datetime.datetime.now()
        with open(local, "rb") as f:
            serialized = f.read()
        print()
        print("Reading file took ", datetime.datetime.now() - start)
        load(serialized)
        print("Deserialization (Safe) took ", datetime.datetime.now() - start)

    def test_gpt2_deserialization_safe_mmap(self):
        filename = hf_hub_download(MODEL_ID, filename="pytorch_model.bin")
        data = torch.load(filename)

        local = "out_safe_mmap.bin"
        save_file(data, local)

        start = datetime.datetime.now()
        load_file(local)
        print()
        print("Deserialization (Safe) took ", datetime.datetime.now() - start)

    def test_gpt2_deserialization_pt(self):
        filename = hf_hub_download(MODEL_ID, filename="pytorch_model.bin")

        start = datetime.datetime.now()
        torch.load(filename)
        print()
        print("Deserialization (PT) took ", datetime.datetime.now() - start)

    def test_gpt2_deserialization_flax(self):
        from flax.serialization import msgpack_restore

        filename = hf_hub_download(MODEL_ID, filename="flax_model.msgpack")
        start = datetime.datetime.now()
        with open(filename, "rb") as f:
            data = f.read()
        msgpack_restore(data)
        print()
        print("Deserialization (flax) took ", datetime.datetime.now() - start)

    def test_gpt2_deserialization_tf(self):
        import h5py
        import numpy as np

        filename = hf_hub_download(MODEL_ID, filename="tf_model.h5")
        start = datetime.datetime.now()
        tensors = {}

        def _load(f, prefix=""):
            for k in f.keys():
                if isinstance(f[k], h5py._hl.dataset.Dataset):
                    tensors[f"{prefix}_{k}"] = np.array(f[k])
                else:
                    _load(f[k], prefix=f"{prefix}_{k}")

        with h5py.File(filename, "r") as f:
            _load(f)

        print()
        print("Deserialization (TF) took ", datetime.datetime.now() - start)

    def test_gpt2_serialization_pt(self):
        filename = hf_hub_download(MODEL_ID, filename="pytorch_model.bin")
        data = torch.load(filename)

        start = datetime.datetime.now()
        with open("out_pt.bin", "wb") as f:
            torch.save(data, f)
        print("Serialization (PT) took ", datetime.datetime.now() - start)

    def test_gpt2_serialization_safe(self):
        filename = hf_hub_download(MODEL_ID, filename="pytorch_model.bin")
        data = torch.load(filename)
        start = datetime.datetime.now()
        outfilename = "out_py_test.bin"
        out = save_file(data, outfilename)
        print("Serialization (safe) took ", datetime.datetime.now() - start)
        del out

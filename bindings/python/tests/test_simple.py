import unittest
from safetensors import save, load, load_pt, save_pt
import numpy as np
from huggingface_hub import hf_hub_download
import torch
import os
import datetime


class TestCase(unittest.TestCase):
    def test_serialization(self):
        data = np.zeros((2, 2), dtype=np.int32)
        out = save({"test": data})

        self.assertEqual(
            out,
            b"""<\x00\x00\x00\x00\x00\x00\x00{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00""",
        )

        data[1, 1] = 1
        out = save({"test": data})

        self.assertEqual(
            out,
            b"""<\x00\x00\x00\x00\x00\x00\x00{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\00""",
        )

    def test_deserialization(self):
        serialized = b"""<\x00\x00\x00\x00\x00\x00\x00{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"""

        out = load(serialized)
        self.assertEqual(list(out.keys()), ["test"])
        np.testing.assert_array_equal(out["test"], np.zeros((2, 2), dtype=np.int32))


class BigTestCase(unittest.TestCase):
    def test_gpt2_deserialization(self):
        start = datetime.datetime.now()
        with open("out_py.bin", "rb") as f:
            serialized = f.read()
        print("Reading file took ", datetime.datetime.now() - start)
        out = load_pt(serialized)
        print("Deserialization took ", datetime.datetime.now() - start)

    def test_gpt2_serialization(self):
        filename = hf_hub_download("gpt2", filename="pytorch_model.bin")
        data = torch.load(filename)

        start = datetime.datetime.now()
        out = save_pt(data)
        with open("out_py_test.bin", "wb") as f:
            f.write(out)
        print("Saving took ", datetime.datetime.now() - start)

        start = datetime.datetime.now()
        with open("out_py_test_pt.bin", "wb") as f:
            torch.save(data, f)
        print("PT Saving took ", datetime.datetime.now() - start)

import os
import unittest
from pathlib import Path

import numpy as np
import torch

from safetensors.numpy import load, load_file, save, save_file
from safetensors import safe_open, SafetensorError, serialize
from safetensors.torch import load_file as load_file_pt
from safetensors.torch import save_file as save_file_pt


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

    def test_serialization_order_invariant(self):
        data = np.zeros((2, 2), dtype=np.int32)
        out1 = save({"test1": data, "test2": data})
        out2 = save({"test2": data, "test1": data})

        self.assertEqual(out1, out2)

    def test_serialization_no_big_endian(self):
        # Big endian tensor
        data = np.zeros((2, 2), dtype=">u4")
        with self.assertRaises(ValueError):
            save({"test1": data})

    def test_accept_path(self):
        tensors = {
            "a": torch.zeros((2, 2)),
            "b": torch.zeros((2, 3), dtype=torch.uint8),
        }
        save_file_pt(tensors, Path("./out.safetensors"))
        load_file_pt(Path("./out.safetensors"))
        os.remove(Path("./out.safetensors"))


class WindowsTestCase(unittest.TestCase):
    def test_get_correctly_dropped(self):
        tensors = {
            "a": torch.zeros((2, 2)),
            "b": torch.zeros((2, 3), dtype=torch.uint8),
        }
        save_file_pt(tensors, "./out.safetensors")
        with safe_open("./out.safetensors", framework="pt") as f:
            pass

        with self.assertRaises(SafetensorError):
            print(f.keys())

        with open("./out.safetensors", "w") as g:
            g.write("something")


class ReadmeTestCase(unittest.TestCase):
    def assertTensorEqual(self, tensors1, tensors2, equality_fn):
        self.assertEqual(tensors1.keys(), tensors2.keys(), "tensor keys don't match")

        for k, v1 in tensors1.items():
            v2 = tensors2[k]

            self.assertTrue(equality_fn(v1, v2), f"{k} tensors are different")

    def test_numpy_example(self):
        tensors = {"a": np.zeros((2, 2)), "b": np.zeros((2, 3), dtype=np.uint8)}

        save_file(tensors, "./out.safetensors")
        out = save(tensors)

        # Now loading
        loaded = load_file("./out.safetensors")
        self.assertTensorEqual(tensors, loaded, np.allclose)

        loaded = load(out)
        self.assertTensorEqual(tensors, loaded, np.allclose)

    def test_torch_example(self):
        tensors = {
            "a": torch.zeros((2, 2)),
            "b": torch.zeros((2, 3), dtype=torch.uint8),
        }
        # Saving modifies the tensors to type numpy, so we must copy for the
        # test to be correct.
        tensors2 = tensors.copy()

        save_file_pt(tensors, "./out.safetensors")

        # Now loading
        loaded = load_file_pt("./out.safetensors")
        self.assertTensorEqual(tensors2, loaded, torch.allclose)

    def test_exception(self):
        flattened = {"test": {"dtype": "float32", "shape": [1]}}

        with self.assertRaises(SafetensorError):
            serialize(flattened)

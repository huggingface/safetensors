import unittest

import numpy as np
import torch

from safetensors.numpy import load, load_file, save, save_file
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


class ReadmeTestCase(unittest.TestCase):
    def assertTensorEqual(self, tensors1, tensors2, equality_fn):
        self.assertEqual(tensors1.keys(), tensors2.keys(), "tensor keys don't match")

        for k, v1 in tensors1.items():
            v2 = tensors2[k]

            self.assertTrue(equality_fn(v1, v2), f"{k} tensors are different")

    def test_numpy_example(self):
        tensors = {"a": np.zeros((2, 2)), "b": np.zeros((2, 3), dtype=np.uint8)}

        save_file(tensors, "./out.safetensors")

        # Now loading
        loaded = load_file("./out.safetensors")
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

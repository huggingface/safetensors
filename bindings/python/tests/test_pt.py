from pathlib import Path
import unittest

import torch

from safetensors.safetensors_rust import safe_open
from safetensors.torch import load, load_file, save, save_file

this_dir = Path(__file__).parent
data_dir = this_dir / "data"

class TorchTestCase(unittest.TestCase):
    PT_MMAP_FILE = str(data_dir / "out_safe_pt_mmap_small.safetensors")  # FixMe: must accept Path!
    MMAP_SERIALIZATION = None
    MMAP_DATA = None

    @classmethod
    def setUpClass(cls):
        cls.MMAP_DATA = {
            "test": torch.zeros((2, 2), dtype=torch.bfloat16),
            "test2": torch.zeros((2, 2), dtype=torch.float16),
            "test3": torch.zeros((2, 2), dtype=torch.bool),
        }
        save_file(cls.MMAP_DATA, cls.PT_MMAP_FILE)
        cls.MMAP_SERIALIZATION = save(cls.MMAP_DATA)

    def verify_loaded_data(self, reloaded):
        self.assertIsInstance(reloaded, dict)
        self.assertTrue(torch.equal(self.__class__.MMAP_DATA["test"], reloaded["test"]))
        self.assertTrue(torch.equal(self.__class__.MMAP_DATA["test2"], reloaded["test2"]))
        self.assertTrue(torch.equal(self.__class__.MMAP_DATA["test3"], reloaded["test3"]))

    def test_odd_dtype(self):
        self.verify_loaded_data(load_file(self.__class__.PT_MMAP_FILE))

    def test_load(self):
        self.verify_loaded_data(load(self.__class__.MMAP_SERIALIZATION))

    @unittest.skipIf(not torch.cuda.is_available(), "Cuda is not available")
    def test_gpu(self):
        data = {
            "test": torch.arange(4).view((2, 2)).to("cuda:0"),
        }
        save_file(data, self.__class__.PT_MMAP_FILE)
        reloaded = load_file(self.__class__.PT_MMAP_FILE)
        self.assertTrue(torch.equal(torch.arange(4).view((2, 2)), reloaded["test"]))

    def test_sparse(self):
        data = {"test": torch.sparse_coo_tensor(size=(2, 3))}
        local = "./tests/data/out_safe_pt_sparse.safetensors"
        with self.assertRaises(ValueError):
            save_file(data, local)


class SliceTestCase(unittest.TestCase):
    def setUp(self):
        self.tensor = torch.arange(6, dtype=torch.float32).reshape((1, 2, 3))
        self.data = {"test": self.tensor}
        self.local = "./tests/data/out_safe_pt_mmap_slice.safetensors"
        # Need to copy since that call mutates the tensors to numpy
        save_file(self.data.copy(), self.local)

    def test_cannot_serialize_a_non_contiguous_tensor(self):
        tensor = torch.arange(6, dtype=torch.float32).reshape((1, 2, 3))
        x = tensor[:, :, 1]
        data = {"test": x}
        self.assertFalse(
            x.is_contiguous(),
        )
        with self.assertRaises(ValueError):
            save_file(data, "./tests/data/out.safetensors")

    def test_deserialization_slice(self):
        with safe_open(self.local, framework="pt") as f:
            tensor = f.get_slice("test")[:, :, 1:2]

        self.assertEqual(
            tensor.numpy().tobytes(),
            b"\x00\x00\x80?\x00\x00\x80@",
        )

        self.assertTrue(torch.equal(tensor, torch.Tensor([[[1.0], [4.0]]])))
        self.assertTrue(torch.equal(tensor, self.tensor[:, :, 1:2]))

    def test_deserialization_metadata(self):
        with safe_open(self.local, framework="pt") as f:
            metadata = f.metadata()
        self.assertEqual(metadata, None)

        # Save another one *with* metadata
        tensor = torch.arange(6, dtype=torch.float32).reshape((1, 2, 3))
        data = {"test": tensor}
        local = "./tests/data/out_safe_pt_mmap2.safetensors"
        # Need to copy since that call mutates the tensors to numpy
        save_file(data, local, metadata={"Something": "more"})
        with safe_open(local, framework="pt") as f:
            metadata = f.metadata()
        self.assertEqual(metadata, {"Something": "more"})

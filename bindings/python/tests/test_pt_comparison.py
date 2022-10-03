import unittest
from safetensors.torch import save_file, load_file
from safetensors.safetensors_rust import safe_open
from huggingface_hub import hf_hub_download
import torch
import datetime
import os


MODEL_ID = os.getenv("MODEL_ID", "gpt2")


class TorchTestCase(unittest.TestCase):
    def test_odd_dtype(self):
        data = {
            "test": torch.zeros((2, 2), dtype=torch.bfloat16),
            "test2": torch.zeros((2, 2), dtype=torch.float16),
            "test3": torch.zeros((2, 2), dtype=torch.bool),
        }
        local = "./tests/data/out_safe_pt_mmap.bin"
        save_file(data, local)
        reloaded = load_file(local)
        self.assertTrue(torch.equal(data["test"], reloaded["test"]))
        self.assertTrue(torch.equal(data["test2"], reloaded["test2"]))
        self.assertTrue(torch.equal(data["test3"], reloaded["test3"]))

    def test_gpu(self):
        data = {
            "test": torch.arange(4).view((2, 2)).to('cuda:0'),
        }
        local = "./tests/data/out_safe_pt_mmap.bin"
        save_file(data, local)
        reloaded = load_file(local)
        self.assertTrue(torch.equal(torch.arange(4).view((2, 2)), reloaded["test"]))

    def test_sparse(self):
        data = {
            "test": torch.sparse_coo_tensor(size=(2, 3))
        }
        local = "./tests/data/out_safe_pt_sparse.bin"
        with self.assertRaises(ValueError):
            save_file(data, local)



class SpeedTestCase(unittest.TestCase):
    def setUp(self):
        self.filename = hf_hub_download(MODEL_ID, filename="pytorch_model.bin")
        self.data = torch.load(self.filename, map_location="cpu")
        self.local = "./tests/data/out_safe_pt_mmap.bin"
        # Need to copy since that call mutates the tensors to numpy
        save_file(self.data.copy(), self.local)

    def test_deserialization_safe(self):
        start = datetime.datetime.now()
        # First time to hit disk
        load_file(self.local)
        # Second time we should be in disk cache
        start = datetime.datetime.now()
        load_file(self.local)
        safe_time = datetime.datetime.now() - start
        torch.load(self.filename)
        start = datetime.datetime.now()
        torch.load(self.filename)
        pt_time = datetime.datetime.now() - start
        print()
        print(f"Deserialization (Safe) took {safe_time}")
        print(f"Deserialization (PT) took {pt_time} (Safe is {pt_time/safe_time} faster)")

    def test_serialization_safe(self):
        start = datetime.datetime.now()
        outfilename = "./tests/data/out_safe.bin"
        save_file(self.data, outfilename)
        safe_time = datetime.datetime.now() - start

        start = datetime.datetime.now()
        with open("./tests/data/out_pt.bin", "wb") as f:
            torch.save(self.data, f)
        pt_time = datetime.datetime.now() - start

        print()
        print(f"Serialization (safe) took {safe_time}")
        print(f"Serialization (PT) took {pt_time} (Safe is {pt_time/safe_time} faster)")


class SliceTestCase(unittest.TestCase):
    def setUp(self):
        self.tensor = torch.arange(6, dtype=torch.float32).reshape((1, 2, 3))
        self.data = {"test": self.tensor}
        self.local = "./tests/data/out_safe_pt_mmap.bin"
        # Need to copy since that call mutates the tensors to numpy
        save_file(self.data.copy(), self.local)

    def test_cannot_serialize_a_non_contiguous_tensor(self):
        tensor = torch.arange(6, dtype=torch.float32).reshape((1, 2, 3))
        x = tensor[:,:, 1]
        data = {"test": x}
        self.assertFalse(x.is_contiguous(), )
        with self.assertRaises(ValueError):
            save_file(data, "./tests/data/out.bin")

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
        local = "./tests/data/out_safe_pt_mmap2.bin"
        # Need to copy since that call mutates the tensors to numpy
        save_file(data, local, metadata={"Something": "more"})
        with safe_open(local, framework="pt") as f:
            metadata = f.metadata()
        self.assertEqual(metadata, {"Something": "more"})

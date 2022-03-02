import unittest
from safetensors.torch import save_file, load_file, load
from huggingface_hub import hf_hub_download
import torch
import datetime
import os


MODEL_ID = os.getenv("MODEL_ID", "gpt2")


class SafeTestCase(unittest.TestCase):
    def setUp(self):
        self.filename = hf_hub_download(MODEL_ID, filename="pytorch_model.bin")
        self.data = torch.load(self.filename, map_location="cpu")
        print("PT keys", len(self.data.keys()))
        self.local = "./tests/data/out_safe_pt_mmap.bin"
        # Need to copy since that call mutates the tensors to numpy
        save_file(self.data.copy(), self.local)

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
        print("Serialization (safe) took ", datetime.datetime.now() - start)


class TorchTestCase(unittest.TestCase):
    def setUp(self):
        self.filename = hf_hub_download(MODEL_ID, filename="pytorch_model.bin")
        self.data = torch.load(self.filename)

    def test_deserialization(self):
        start = datetime.datetime.now()
        torch.load(self.filename)
        print()
        print("Deserialization (PT) took ", datetime.datetime.now() - start)

    def test_serialization(self):
        start = datetime.datetime.now()
        with open("./tests/data/out_pt.bin", "wb") as f:
            torch.save(self.data, f)
        print("Serialization (PT) took ", datetime.datetime.now() - start)

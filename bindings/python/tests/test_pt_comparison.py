import unittest
from safetensors import save_file_pt, load_file, np2pt, load
from huggingface_hub import hf_hub_download
import torch
import datetime


class BigTestCase(unittest.TestCase):
    def test_gpt2_deserialization_safe(self):
        filename = hf_hub_download("gpt2", filename="pytorch_model.bin")
        data = torch.load(filename)

        local = "out_safe.bin"
        save_file_pt(data, local)

        start = datetime.datetime.now()
        with open(local, "rb") as f:
            serialized = f.read()
        print()
        print("Reading file took ", datetime.datetime.now() - start)
        np2pt(load(serialized))
        print("Deserialization (Safe) took ", datetime.datetime.now() - start)

    def test_gpt2_deserialization_safe_mmap(self):
        filename = hf_hub_download("gpt2", filename="pytorch_model.bin")
        data = torch.load(filename)

        local = "out_safe_mmap.bin"
        save_file_pt(data, local)

        start = datetime.datetime.now()
        np2pt(load_file(local))
        print()
        print("Deserialization (Safe) took ", datetime.datetime.now() - start)

    def test_gpt2_deserialization_pt(self):
        filename = hf_hub_download("gpt2", filename="pytorch_model.bin")

        start = datetime.datetime.now()
        torch.load(filename)
        print()
        print("Deserialization (PT) took ", datetime.datetime.now() - start)

    def test_gpt2_serialization_pt(self):
        filename = hf_hub_download("gpt2", filename="pytorch_model.bin")
        data = torch.load(filename)

        start = datetime.datetime.now()
        with open("out_pt.bin", "wb") as f:
            torch.save(data, f)
        print("Serialization (PT) took ", datetime.datetime.now() - start)

    def test_gpt2_serialization_safe(self):
        filename = hf_hub_download("gpt2", filename="pytorch_model.bin")
        data = torch.load(filename)
        start = datetime.datetime.now()
        outfilename = "out_py_test.bin"
        out = save_file_pt(data, outfilename)
        print("Serialization (safe) took ", datetime.datetime.now() - start)
        del out

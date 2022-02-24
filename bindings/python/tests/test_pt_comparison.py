import unittest
from safetensors import load_pt, save_pt
from huggingface_hub import hf_hub_download
import torch
import datetime


class BigTestCase(unittest.TestCase):
    def test_gpt2_deserialization_safe(self):
        filename = hf_hub_download("gpt2", filename="pytorch_model.bin")
        local = "out_safe.bin"
        data = torch.load(filename)

        out = save_pt(data)
        with open(local, "wb") as f:
            f.write(out)

        start = datetime.datetime.now()
        with open(local, "rb") as f:
            serialized = f.read()
        print()
        print("Reading file took ", datetime.datetime.now() - start)
        load_pt(serialized)
        print("Deserialization (Safe) took ", datetime.datetime.now() - start)

    def test_gpt2_deserialization_pt(self):
        filename = hf_hub_download("gpt2", filename="pytorch_model.bin")

        start = datetime.datetime.now()
        torch.load(filename)
        print()
        print("Deserialization (PT) took ", datetime.datetime.now() - start)

    def test_gpt2_serialization(self):
        filename = hf_hub_download("gpt2", filename="pytorch_model.bin")
        data = torch.load(filename)

        start = datetime.datetime.now()
        out = save_pt(data)
        with open("out_py_test.bin", "wb") as f:
            f.write(out)
        print()
        print("Serialization (safe) took ", datetime.datetime.now() - start)
        del out

        start = datetime.datetime.now()
        with open("out_pt.bin", "wb") as f:
            torch.save(data, f)
        print("Serialization (PT) took ", datetime.datetime.now() - start)

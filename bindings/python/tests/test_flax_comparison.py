import unittest
from safetensors.flax import save_file, load_file
from huggingface_hub import hf_hub_download
from flax.serialization import msgpack_restore, msgpack_serialize
import numpy as np
import jax.numpy as jnp
import datetime
import os

MODEL_ID = os.getenv("MODEL_ID", "gpt2")


def _load(nested, flat, prefix=""):
    for k, v in nested.items():
        if isinstance(v, dict):
            _load(v, flat, prefix=f"{prefix}_{k}")
        elif isinstance(v, np.ndarray):
            flat[f"{prefix}_{k}"] = jnp.array(v)


def load(nested_np_dicts):
    tensors = {}
    _load(nested_np_dicts, tensors)
    return tensors


class SafeTestCase(unittest.TestCase):
    def setUp(self):
        self.filename = hf_hub_download(MODEL_ID, filename="flax_model.msgpack")
        with open(self.filename, "rb") as f:
            data = f.read()
        self.data = load(msgpack_restore(data))
        self.local = "./tests/data/out_safe_flax_mmap.bin"
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
        print("Serialization (safe) took ", datetime.datetime.now() - start)


class FlaxTestCase(unittest.TestCase):
    def setUp(self):
        self.filename = hf_hub_download(MODEL_ID, filename="flax_model.msgpack")
        with open(self.filename, "rb") as f:
            data = f.read()
        self.data = msgpack_restore(data)

    def test_deserialization_flax(self):
        start = datetime.datetime.now()
        with open(self.filename, "rb") as f:
            data = f.read()
        msgpack_restore(data)
        print()
        print("Deserialization (flax) took ", datetime.datetime.now() - start)

    def test_serialization_flax(self):
        start = datetime.datetime.now()
        serialized = msgpack_serialize(self.data)
        with open("./tests/data/out_flax.msgpack", "wb") as f:
            f.write(serialized)
        print()
        print("Serialization (flax) took ", datetime.datetime.now() - start)

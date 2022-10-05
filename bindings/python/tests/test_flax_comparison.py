import datetime
import os
import platform
import unittest

import numpy as np

from huggingface_hub import hf_hub_download


if platform.system() != "Windows":
    # This platform is not supported, we don't want to crash on import
    # This test will be skipped anyway.
    import jax.numpy as jnp
    from flax.serialization import msgpack_restore, msgpack_serialize
    from safetensors.flax import load_file, save_file

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


# Jax doesn't not exist on Windows
@unittest.skipIf(platform.system() == "Windows", "Jax is not available on Windows")
class SafeTestCase(unittest.TestCase):
    def setUp(self):
        self.filename = hf_hub_download(MODEL_ID, filename="flax_model.msgpack")
        with open(self.filename, "rb") as f:
            data = f.read()
        self.data = load(msgpack_restore(data))
        self.local = "./tests/data/out_safe_flax_mmap.bin"
        save_file(self.data, self.local)

    def test_deserialization_safe(self):
        load_file(self.local)

        start = datetime.datetime.now()
        load_file(self.local)
        safe_time = datetime.datetime.now() - start

        with open(self.filename, "rb") as f:
            data = f.read()

        start = datetime.datetime.now()
        with open(self.filename, "rb") as f:
            data = f.read()
        msgpack_restore(data)
        flax_time = datetime.datetime.now() - start

        print()
        print(f"Deserialization (Safe) took {safe_time}")
        print(f"Deserialization (flax) took {flax_time} (Safe is {flax_time/safe_time} faster)")

    def test_serialization_safe(self):
        outfilename = "./tests/data/out_safe.bin"
        save_file(self.data, outfilename)
        start = datetime.datetime.now()
        save_file(self.data, outfilename)
        safe_time = datetime.datetime.now() - start
        start = datetime.datetime.now()
        serialized = msgpack_serialize(self.data)
        with open("./tests/data/out_flax.msgpack", "wb") as f:
            f.write(serialized)
        flax_time = datetime.datetime.now() - start
        print()
        print(f"Serialization (safe) took {safe_time}")
        print("Serialization (flax) took ", datetime.datetime.now() - start)
        print(f"Deserialization (flax) took {flax_time} (Safe is {flax_time/safe_time} faster)")

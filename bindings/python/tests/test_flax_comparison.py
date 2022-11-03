import platform
import unittest

import numpy as np


if platform.system() != "Windows":
    # This platform is not supported, we don't want to crash on import
    # This test will be skipped anyway.
    import jax.numpy as jnp
    from flax.serialization import msgpack_restore, msgpack_serialize
    from safetensors import safe_open
    from safetensors.flax import load_file, save_file


# Jax doesn't not exist on Windows
@unittest.skipIf(platform.system() == "Windows", "Flax is not available on Windows")
class LoadTestCase(unittest.TestCase):
    def setUp(self):
        data = {
            "test": jnp.zeros((1024, 1024), dtype=jnp.float32),
            "test2": jnp.zeros((1024, 1024), dtype=jnp.float32),
            "test3": jnp.zeros((1024, 1024), dtype=jnp.float32),
        }
        self.flax_filename = "./tests/data/flax_load.msgpack"
        self.sf_filename = "./tests/data/flax_load.safetensors"

        serialized = msgpack_serialize(data)
        with open(self.flax_filename, "wb") as f:
            f.write(serialized)

        save_file(data, self.sf_filename)

    def test_deserialization_safe(self):
        weights = load_file(self.sf_filename)

        with open(self.flax_filename, "rb") as f:
            data = f.read()
        flax_weights = msgpack_restore(data)

        for k, v in weights.items():
            tv = flax_weights[k]
            self.assertTrue(np.allclose(v, tv))

    def test_deserialization_safe_open(self):
        weights = {}
        with safe_open(self.sf_filename, framework="flax") as f:
            for k in f.keys():
                weights[k] = f.get_tensor(k)

        with open(self.flax_filename, "rb") as f:
            data = f.read()
        flax_weights = msgpack_restore(data)

        for k, v in weights.items():
            tv = flax_weights[k]
            self.assertTrue(np.allclose(v, tv))

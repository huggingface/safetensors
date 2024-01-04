import platform
import unittest


if platform.system() == "Darwin":
    # This platform is not supported, we don't want to crash on import
    # This test will be skipped anyway.
    import mlx.core as mx
    from safetensors import safe_open
    from safetensors.mlx import load_file, save_file


# Jax doesn't not exist on Windows
@unittest.skipIf(platform.system() != "Darwin", "Mlx is not available on non Mac")
class LoadTestCase(unittest.TestCase):
    def setUp(self):
        data = {
            "test": mx.zeros((1024, 1024), dtype=mx.float32),
            "test2": mx.zeros((1024, 1024), dtype=mx.float32),
            "test3": mx.zeros((1024, 1024), dtype=mx.float32),
            "test4": mx.zeros((1024, 1024), dtype=mx.bfloat16),
        }
        self.mlx_filename = "./tests/data/mlx_load.npz"
        self.sf_filename = "./tests/data/mlx_load.safetensors"

        serialized = mx.savez(self.mlx_filename, **data)
        save_file(data, self.sf_filename)

    def test_zero_sized(self):
        data = {
            "test": mx.zeros((2, 0), dtype=mx.float32),
        }
        local = "./tests/data/out_safe_flat_mmap_small2.safetensors"
        save_file(data.copy(), local)
        reloaded = load_file(local)
        # Empty tensor != empty tensor on numpy, so comparing shapes
        # instead
        self.assertEqual(data["test"].shape, reloaded["test"].shape)

    def test_deserialization_safe(self):
        weights = load_file(self.sf_filename)

        mlx_weights = mx.load(self.mlx_filename)

        for k, v in weights.items():
            tv = mlx_weights[k]
            self.assertTrue(mx.allclose(v, tv))

    def test_deserialization_safe_open(self):
        weights = {}
        with safe_open(self.sf_filename, framework="mlx") as f:
            for k in f.keys():
                weights[k] = f.get_tensor(k)

        mlx_weights = mx.load(self.mlx_filename)

        for k, v in weights.items():
            tv = mlx_weights[k]
            self.assertTrue(mx.allclose(v, tv))

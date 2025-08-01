import unittest

import numpy as np

from safetensors import safe_open


try:
    import paddle
    from safetensors.paddle import load_file, save_file

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False


@unittest.skipIf(not HAS_PADDLE, "Paddle is not available")
class SafeTestCase(unittest.TestCase):
    def setUp(self):
        data = {
            "test": paddle.zeros((1024, 1024), dtype=paddle.float32),
            "test2": paddle.zeros((1024, 1024), dtype=paddle.float32),
            "test3": paddle.zeros((1024, 1024), dtype=paddle.float32),
        }
        self.paddle_filename = "./tests/data/paddle_load.pdparams"
        self.sf_filename = "./tests/data/paddle_load.safetensors"

        paddle.save(data, self.paddle_filename)
        save_file(data, self.sf_filename)

    @unittest.expectedFailure
    def test_zero_sized(self):
        # This fails because paddle wants initialized tensor before
        # sending to numpy
        data = {
            "test": paddle.zeros((2, 0), dtype=paddle.float32),
        }
        local = "./tests/data/out_safe_paddle_mmap_small2.safetensors"
        save_file(data, local)
        reloaded = load_file(local)
        self.assertTrue(paddle.equal(data["test"], reloaded["test"]))

    def test_deserialization_safe(self):
        weights = load_file(self.sf_filename)

        paddle_weights = paddle.load(self.paddle_filename)
        for k, v in weights.items():
            tv = paddle_weights[k]
            self.assertTrue(np.allclose(v, tv))


@unittest.skipIf(not HAS_PADDLE, "Paddle is not available")
class WithOpenCase(unittest.TestCase):
    def test_paddle_tensor_cpu(self):
        A = paddle.randn((10, 5))
        tensors = {
            "a": A,
        }
        save_file(tensors, f"./tensor_paddle.safetensors")

        # Now loading cpu
        with safe_open(f"./tensor_paddle.safetensors", framework="pp", device="cpu") as f:
            tensor = f.get_tensor("a")
            self.assertEqual(list(tensor.shape), [10, 5])
            assert paddle.allclose(tensor, A).item()
            assert not tensor.place.is_gpu_place()

    def test_paddle_tensor_gpu(self):
        A = paddle.randn((10, 5))
        tensors = {
            "a": A,
        }
        save_file(tensors, f"./tensor_paddle.safetensors")
        # Now loading gpu
        with safe_open(f"./tensor_paddle.safetensors", framework="pp", device="cuda") as f:
            tensor = f.get_tensor("a")
            self.assertEqual(list(tensor.shape), [10, 5])
            assert paddle.allclose(tensor, A).item()
            assert tensor.place.is_gpu_place()

    def test_paddle_slice_cpu(self):
        A = paddle.randn((10, 5))
        tensors = {
            "a": A,
        }
        save_file(tensors, f"./slice_paddle.safetensors")

        # Now loading
        with safe_open(f"./slice_paddle.safetensors", framework="pp", device="cpu") as f:
            slice_ = f.get_slice("a")
            tensor = slice_[:]
            self.assertEqual(list(tensor.shape), [10, 5])
            assert paddle.allclose(tensor, A).item()
            assert not tensor.place.is_gpu_place()

            tensor = slice_[tuple()]
            self.assertEqual(list(tensor.shape), [10, 5])
            assert paddle.allclose(tensor, A).item()
            assert not tensor.place.is_gpu_place()

            tensor = slice_[:2]
            self.assertEqual(list(tensor.shape), [2, 5])
            assert paddle.allclose(tensor, A[:2]).item()
            assert not tensor.place.is_gpu_place()

            tensor = slice_[:, :2]
            self.assertEqual(list(tensor.shape), [10, 2])
            assert paddle.allclose(tensor, A[:, :2]).item()
            assert not tensor.place.is_gpu_place()

            tensor = slice_[0, :2]
            self.assertEqual(list(tensor.shape), [2])
            assert paddle.allclose(tensor, A[0, :2]).item()
            assert not tensor.place.is_gpu_place()

            tensor = slice_[2:, 0]
            self.assertEqual(list(tensor.shape), [8])
            assert paddle.allclose(tensor, A[2:, 0]).item()
            assert not tensor.place.is_gpu_place()

            tensor = slice_[2:, 1]
            self.assertEqual(list(tensor.shape), [8])
            assert paddle.allclose(tensor, A[2:, 1]).item()
            assert not tensor.place.is_gpu_place()

            tensor = slice_[2:, -1]
            self.assertEqual(list(tensor.shape), [8])
            assert paddle.allclose(tensor, A[2:, -1]).item()
            assert not tensor.place.is_gpu_place()

            tensor = slice_[list()]
            self.assertEqual(list(tensor.shape), [0, 5])
            assert paddle.allclose(tensor, A[list()]).item()
            assert not tensor.place.is_gpu_place()

    def test_paddle_slice_gpu(self):
        A = paddle.randn((10, 5))
        tensors = {
            "a": A,
        }
        save_file(tensors, f"./slice_paddle.safetensors")

        # Now loading
        with safe_open(f"./slice_paddle.safetensors", framework="pp", device="cuda") as f:
            slice_ = f.get_slice("a")
            tensor = slice_[:]
            self.assertEqual(list(tensor.shape), [10, 5])
            assert paddle.allclose(tensor, A).item()
            assert tensor.place.is_gpu_place()

            tensor = slice_[tuple()]
            self.assertEqual(list(tensor.shape), [10, 5])
            assert paddle.allclose(tensor, A).item()
            assert tensor.place.is_gpu_place()

            tensor = slice_[:2]
            self.assertEqual(list(tensor.shape), [2, 5])
            assert paddle.allclose(tensor, A[:2]).item()
            assert tensor.place.is_gpu_place()

            tensor = slice_[:, :2]
            self.assertEqual(list(tensor.shape), [10, 2])
            assert paddle.allclose(tensor, A[:, :2]).item()
            assert tensor.place.is_gpu_place()

            tensor = slice_[0, :2]
            self.assertEqual(list(tensor.shape), [2])
            assert paddle.allclose(tensor, A[0, :2]).item()
            assert tensor.place.is_gpu_place()

            tensor = slice_[2:, 0]
            self.assertEqual(list(tensor.shape), [8])
            assert paddle.allclose(tensor, A[2:, 0]).item()
            assert tensor.place.is_gpu_place()

            tensor = slice_[2:, 1]
            self.assertEqual(list(tensor.shape), [8])
            assert paddle.allclose(tensor, A[2:, 1]).item()
            assert tensor.place.is_gpu_place()

            tensor = slice_[2:, -1]
            self.assertEqual(list(tensor.shape), [8])
            assert paddle.allclose(tensor, A[2:, -1]).item()
            assert tensor.place.is_gpu_place()

            tensor = slice_[list()]
            self.assertEqual(list(tensor.shape), [0, 5])
            assert paddle.allclose(tensor, A[list()]).item()
            assert tensor.place.is_gpu_place()
import unittest

import numpy as np

from safetensors import _safe_open_handle
from safetensors.numpy import save_file, save


class ReadmeTestCase(unittest.TestCase):
    def assertTensorEqual(self, tensors1, tensors2, equality_fn):
        self.assertEqual(tensors1.keys(), tensors2.keys(), "tensor keys don't match")

        for k, v1 in tensors1.items():
            v2 = tensors2[k]

            self.assertTrue(equality_fn(v1, v2), f"{k} tensors are different")

    def test_numpy_example(self):
        tensors = {"a": np.zeros((2, 2)), "b": np.zeros((2, 3), dtype=np.uint8)}

        save_file(tensors, "./out_np.safetensors")

        # Now loading
        loaded = {}
        with open("./out_np.safetensors", "r") as f:
            with safe_open_handle(f, framework="np", device="cpu") as g:
                for key in g.keys():
                    loaded[key] = g.get_tensor(key)
        self.assertTensorEqual(tensors, loaded, np.allclose)

    def test_fsspec(self):
        import fsspec

        tensors = {"a": np.zeros((2, 2)), "b": np.zeros((2, 3), dtype=np.uint8)}

        fs = fsspec.filesystem("file")
        byts = save(tensors)
        with fs.open("fs.safetensors", "wb") as f:
            f.write(byts)
        # Now loading
        loaded = {}
        with fs.open("fs.safetensors", "rb") as f:
            with safe_open_handle(f, framework="np", device="cpu") as g:
                for key in g.keys():
                    loaded[key] = g.get_tensor(key)
        self.assertTensorEqual(tensors, loaded, np.allclose)

    @unittest.skip("Will not work without s3 access")
    def test_fsspec_s3(self):
        import s3fs

        tensors = {"a": np.zeros((2, 2)), "b": np.zeros((2, 3), dtype=np.uint8)}

        s3 = s3fs.S3FileSystem(anon=True)
        byts = save(tensors)
        print(s3.ls("my-bucket"))
        with s3.open("out/fs.safetensors", "wb") as f:
            f.write(byts)
        # Now loading
        loaded = {}
        with s3.open("out/fs.safetensors", "rb") as f:
            with safe_open_handle(f, framework="np", device="cpu") as g:
                for key in g.keys():
                    loaded[key] = g.get_tensor(key)
        self.assertTensorEqual(tensors, loaded, np.allclose)

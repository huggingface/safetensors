import unittest

import numpy as np

import paddle
from safetensors import safe_open
from safetensors.paddle import load_file, save_file


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

    def test_deserialization_safe(self):
        weights = load_file(self.sf_filename)

        paddle_weights = paddle.load(self.paddle_filename)
        for k, v in weights.items():
            tv = paddle_weights[k]
            self.assertTrue(np.allclose(v, tv))

    def test_deserialization_safe_open(self):
        weights = {}
        with safe_open(self.sf_filename, framework="paddle") as f:
            for k in f.keys():
                weights[k] = f.get_tensor(k)

        paddle_weights = paddle.load(self.paddle_filename)

        for k, v in weights.items():
            tv = paddle_weights[k]
            self.assertTrue(np.allclose(v, tv))

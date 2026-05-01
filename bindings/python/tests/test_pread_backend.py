"""Tests for `safe_open(..., backend="pread")`.

The `pread` backend serves each tensor via `pread(2)` instead of mmap'ing
the file, dropping each host buffer immediately after the device transfer so
cumulative host residency stays bounded at one tensor.
"""

import os
import struct
import tempfile
import unittest

import numpy as np
import torch

from safetensors import safe_open
from safetensors.torch import load_file as load_file_pt
from safetensors.torch import load_model, save_file, save_model


SOURCE_TENSORS = {
    "fp32_2d": torch.arange(12, dtype=torch.float32).reshape(3, 4).contiguous(),
    "bf16_2d": torch.arange(8, dtype=torch.bfloat16).reshape(2, 4).contiguous(),
    "fp16_3d": torch.arange(24, dtype=torch.float16).reshape(2, 3, 4).contiguous(),
    "scalar_fp32": torch.tensor(7.5, dtype=torch.float32),
    "empty_2d": torch.empty((0, 5), dtype=torch.float16),
    "i64_1d": torch.arange(5, dtype=torch.int64),
}

if hasattr(torch, "float8_e4m3fn"):
    SOURCE_TENSORS["fp8_e4m3fn"] = torch.zeros(8, dtype=torch.float8_e4m3fn)


def _tensors_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    # torch.equal is not implemented for sub-byte / fp8 dtypes on some builds,
    # so reinterpret the underlying storage as uint8 and compare bytes.
    try:
        return torch.equal(a, b)
    except RuntimeError:
        return torch.equal(
            a.contiguous().view(torch.uint8),
            b.contiguous().view(torch.uint8),
        )


class PreadBackendTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, "tiny.safetensors")
        save_file(SOURCE_TENSORS, self.path, metadata={"foo": "bar"})

    def tearDown(self):
        self.tempdir.cleanup()

    def _assert_state_dict_equal(self, sd):
        self.assertEqual(set(sd.keys()), set(SOURCE_TENSORS.keys()))
        for k, expected in SOURCE_TENSORS.items():
            actual = sd[k]
            self.assertEqual(actual.dtype, expected.dtype, k)
            self.assertEqual(tuple(actual.shape), tuple(expected.shape), k)
            if expected.numel() > 0:
                self.assertTrue(_tensors_equal(actual.cpu(), expected.cpu()), k)

    def test_safe_open_round_trip(self):
        with safe_open(self.path, framework="pt", device="cpu", backend="pread") as f:
            self.assertEqual(f.metadata(), {"foo": "bar"})
            sd = {k: f.get_tensor(k) for k in f.keys()}
        self._assert_state_dict_equal(sd)

    def test_get_tensors_round_trip(self):
        with safe_open(self.path, framework="pt", device="cpu", backend="pread") as f:
            sd = f.get_tensors()
        self._assert_state_dict_equal(sd)

    def test_load_file_round_trip(self):
        sd = load_file_pt(self.path, backend="pread")
        self._assert_state_dict_equal(sd)

    def test_default_backend_unchanged(self):
        # mmap and pread must produce identical bytes.
        with safe_open(self.path, framework="pt", device="cpu") as f:
            sd_mmap = f.get_tensors()
        with safe_open(self.path, framework="pt", device="cpu", backend="pread") as f:
            sd_pread = f.get_tensors()
        self.assertEqual(set(sd_mmap.keys()), set(sd_pread.keys()))
        for k in sd_mmap:
            a, b = sd_mmap[k], sd_pread[k]
            self.assertEqual(a.dtype, b.dtype, k)
            self.assertEqual(tuple(a.shape), tuple(b.shape), k)
            if a.numel() > 0:
                self.assertTrue(_tensors_equal(a.cpu(), b.cpu()), k)

    def test_get_slice(self):
        with safe_open(self.path, framework="pt", device="cpu", backend="pread") as f:
            slice_obj = f.get_slice("fp32_2d")
            self.assertEqual(list(slice_obj.get_shape()), [3, 4])
            sub = slice_obj[:, 1:3]
        expected = SOURCE_TENSORS["fp32_2d"][:, 1:3]
        self.assertEqual(sub.dtype, expected.dtype)
        self.assertEqual(tuple(sub.shape), tuple(expected.shape))
        self.assertTrue(torch.equal(sub, expected))

    def test_load_model(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 3)

        src = Tiny()
        path = os.path.join(self.tempdir.name, "tiny_model.safetensors")
        save_model(src, path)

        dst = Tiny()
        self.assertFalse(torch.equal(src.lin.weight, dst.lin.weight))
        load_model(dst, path, backend="pread")
        self.assertTrue(torch.equal(src.lin.weight, dst.lin.weight))
        self.assertTrue(torch.equal(src.lin.bias, dst.lin.bias))

    def test_invalid_backend_string_raises(self):
        with self.assertRaises(Exception):
            safe_open(self.path, framework="pt", device="cpu", backend="not_a_backend")

    def test_truncated_header_is_rejected(self):
        bad_path = os.path.join(self.tempdir.name, "truncated.safetensors")
        with open(self.path, "rb") as src, open(bad_path, "wb") as dst:
            head = src.read(8)
            dst.write(head)
            n = struct.unpack("<Q", head)[0]
            dst.write(src.read(n // 2))
        with self.assertRaises(Exception):
            with safe_open(
                bad_path, framework="pt", device="cpu", backend="pread"
            ) as f:
                _ = f.get_tensors()

    def test_data_offset_overflow_is_rejected(self):
        bad_path = os.path.join(self.tempdir.name, "lying.safetensors")
        header = (
            b'{"liar":{"dtype":"F32","shape":[1024,1024],"data_offsets":[0,4194304]}}'
        )
        with open(bad_path, "wb") as f:
            f.write(struct.pack("<Q", len(header)))
            f.write(header)
            f.write(b"\x00")
        with self.assertRaises(Exception):
            with safe_open(
                bad_path, framework="pt", device="cpu", backend="pread"
            ) as f:
                _ = f.get_tensors()

    def test_numpy_framework(self):
        np_path = os.path.join(self.tempdir.name, "np.safetensors")
        from safetensors.numpy import save_file as save_np

        np_data = {
            "a": np.arange(6, dtype=np.float32).reshape(2, 3),
            "b": np.arange(8, dtype=np.int64).reshape(4, 2),
        }
        save_np(np_data, np_path)

        with safe_open(np_path, framework="numpy", backend="pread") as f:
            for k, expected in np_data.items():
                got = f.get_tensor(k)
                self.assertEqual(got.dtype, expected.dtype, k)
                self.assertEqual(got.shape, expected.shape, k)
                np.testing.assert_array_equal(got, expected)


if __name__ == "__main__":
    unittest.main()

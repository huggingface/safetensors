"""Tests for `safe_open(...).get_strided_slice(name, dim, intervals)`.

Equivalence with `torch.cat([f.get_slice(name)[..., a:b, ...] for (a, b) in intervals], dim=dim)`
across the mmap and pread backends, plus input validation.
"""

import os
import tempfile
import unittest

import torch

from safetensors import safe_open
from safetensors.torch import save_file


SOURCE_TENSORS = {
    # Even shapes so f4-aligned intervals would also fit if we add fp4 later.
    "w_2d": torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16).contiguous(),
    "w_3d": torch.arange(4 * 6 * 8, dtype=torch.float32).reshape(4, 6, 8).contiguous(),
    "w_bf16": torch.arange(8 * 16, dtype=torch.bfloat16).reshape(8, 16).contiguous(),
}


def _reference_strided_slice(
    tensor: torch.Tensor, dim: int, intervals: list[tuple[int, int]]
) -> torch.Tensor:
    pieces = []
    for a, b in intervals:
        idx = [slice(None)] * tensor.ndim
        idx[dim] = slice(a, b)
        pieces.append(tensor[tuple(idx)])
    return torch.cat(pieces, dim=dim)


class StridedSliceTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tempdir.name, "tiny.safetensors")
        save_file(SOURCE_TENSORS, self.path)

    def tearDown(self):
        self.tempdir.cleanup()

    def _check(self, name, dim, intervals, backend):
        with safe_open(self.path, framework="pt", device="cpu", backend=backend) as f:
            got = f.get_strided_slice(name, dim=dim, intervals=intervals)
        expected = _reference_strided_slice(SOURCE_TENSORS[name], dim, intervals)
        self.assertEqual(got.dtype, expected.dtype, (name, dim, intervals, backend))
        self.assertEqual(
            tuple(got.shape), tuple(expected.shape), (name, dim, intervals, backend)
        )
        # Compare via uint8 view because bf16 doesn't support `equal` on all builds.
        self.assertTrue(
            torch.equal(
                got.contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            ),
            f"mismatch for {name=} {dim=} {intervals=} {backend=}",
        )

    def test_dim0_two_intervals(self):
        # Like _StridedShard splitting a fused weight on the leading dim.
        for backend in ("mmap", "pread"):
            self._check("w_2d", dim=0, intervals=[(0, 2), (4, 6)], backend=backend)

    def test_dim1_two_intervals(self):
        for backend in ("mmap", "pread"):
            self._check("w_2d", dim=1, intervals=[(0, 4), (8, 12)], backend=backend)

    def test_dim_last_three_intervals(self):
        for backend in ("mmap", "pread"):
            self._check(
                "w_3d", dim=2, intervals=[(0, 2), (3, 5), (6, 8)], backend=backend
            )

    def test_middle_dim_3d(self):
        for backend in ("mmap", "pread"):
            self._check("w_3d", dim=1, intervals=[(0, 1), (3, 5)], backend=backend)

    def test_single_interval_matches_plain_slice(self):
        for backend in ("mmap", "pread"):
            self._check("w_2d", dim=0, intervals=[(2, 5)], backend=backend)

    def test_full_range_single_interval(self):
        for backend in ("mmap", "pread"):
            self._check("w_2d", dim=1, intervals=[(0, 16)], backend=backend)

    def test_bf16_dtype(self):
        for backend in ("mmap", "pread"):
            self._check("w_bf16", dim=0, intervals=[(1, 3), (5, 7)], backend=backend)

    def test_interleaved_unsorted_intervals(self):
        # Caller controls strip order; output is the cat in input order.
        for backend in ("mmap", "pread"):
            self._check("w_2d", dim=0, intervals=[(4, 6), (0, 2)], backend=backend)

    # --- validation ---

    def test_invalid_dim(self):
        with safe_open(self.path, framework="pt", device="cpu") as f:
            with self.assertRaises(Exception):
                f.get_strided_slice("w_2d", dim=5, intervals=[(0, 1)])

    def test_empty_intervals_list_rejected(self):
        with safe_open(self.path, framework="pt", device="cpu") as f:
            with self.assertRaises(Exception):
                f.get_strided_slice("w_2d", dim=0, intervals=[])

    def test_reversed_interval_rejected(self):
        with safe_open(self.path, framework="pt", device="cpu") as f:
            with self.assertRaises(Exception):
                f.get_strided_slice("w_2d", dim=0, intervals=[(4, 2)])

    def test_empty_interval_rejected(self):
        with safe_open(self.path, framework="pt", device="cpu") as f:
            with self.assertRaises(Exception):
                f.get_strided_slice("w_2d", dim=0, intervals=[(3, 3)])

    def test_out_of_range_interval_rejected(self):
        with safe_open(self.path, framework="pt", device="cpu") as f:
            with self.assertRaises(Exception):
                f.get_strided_slice("w_2d", dim=0, intervals=[(0, 100)])

    def test_unknown_tensor_rejected(self):
        with safe_open(self.path, framework="pt", device="cpu") as f:
            with self.assertRaises(Exception):
                f.get_strided_slice("does_not_exist", dim=0, intervals=[(0, 1)])

    def test_overlap_allowed_and_duplicates(self):
        # Overlap is allowed and produces duplicated data — matches torch.cat
        # of overlapping slices.
        for backend in ("mmap", "pread"):
            self._check("w_2d", dim=0, intervals=[(0, 3), (1, 4)], backend=backend)


if __name__ == "__main__":
    unittest.main()

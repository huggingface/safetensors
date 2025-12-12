import os
import tempfile
import unittest

import numpy as np
import torch
from safetensors.torch import save_file, safe_open


class TestGDS(unittest.TestCase):
    """Test GPU Direct Storage functionality."""

    @classmethod
    def setUpClass(cls):
        """Check if CUDA is available."""
        cls.cuda_available = torch.cuda.is_available()
        if not cls.cuda_available:
            print("WARNING: CUDA not available, skipping GDS tests")

    def setUp(self):
        """Create a temporary test file."""
        self.test_file = tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors")
        self.test_file.close()
        
        # Create test tensors
        self.tensors = {
            "tensor_small": torch.randn(10, 20),
            "tensor_medium": torch.randn(100, 200),
            "tensor_large": torch.randn(1000, 2000),
        }
        
        # Save test tensors
        save_file(self.tensors, self.test_file.name)

    def tearDown(self):
        """Remove temporary test file."""
        if os.path.exists(self.test_file.name):
            os.unlink(self.test_file.name)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_gds_basic_loading(self):
        """Test basic GDS tensor loading to GPU."""
        with safe_open(self.test_file.name, framework="pt", device="cuda:0", use_gds=True) as f:
            # Load a tensor using GDS
            tensor = f.get_tensor("tensor_medium")
            
            # Verify it's on the correct device
            self.assertEqual(tensor.device.type, "cuda")
            self.assertEqual(tensor.device.index, 0)
            
            # Verify shape
            self.assertEqual(tensor.shape, self.tensors["tensor_medium"].shape)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_gds_correctness(self):
        """Test that GDS loading produces correct results."""
        # Load with standard method
        with safe_open(self.test_file.name, framework="pt", device="cuda:0", use_gds=False) as f:
            tensor_standard = f.get_tensor("tensor_medium")
        
        # Load with GDS
        with safe_open(self.test_file.name, framework="pt", device="cuda:0", use_gds=True) as f:
            tensor_gds = f.get_tensor("tensor_medium")
        
        # Compare results
        self.assertTrue(
            torch.allclose(tensor_standard, tensor_gds, rtol=1e-5, atol=1e-7),
            "GDS and standard loading should produce identical results"
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_gds_multiple_tensors(self):
        """Test loading multiple tensors with GDS."""
        with safe_open(self.test_file.name, framework="pt", device="cuda:0", use_gds=True) as f:
            for name in ["tensor_small", "tensor_medium", "tensor_large"]:
                tensor = f.get_tensor(name)
                self.assertEqual(tensor.device.type, "cuda")
                self.assertEqual(tensor.shape, self.tensors[name].shape)

    def test_gds_cpu_device_error(self):
        """Test that GDS with CPU device raises an error."""
        with self.assertRaises(Exception) as context:
            with safe_open(self.test_file.name, framework="pt", device="cpu", use_gds=True) as f:
                pass
        
        self.assertIn("CUDA device", str(context.exception))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_gds_slicing_not_supported(self):
        """Test that tensor slicing raises an error with GDS."""
        with safe_open(self.test_file.name, framework="pt", device="cuda:0", use_gds=True) as f:
            with self.assertRaises(Exception) as context:
                # Attempt to slice - get_slice returns a slice object, need to access it
                slice_obj = f.get_slice("tensor_medium")
                # Accessing the slice should fail
                tensor = slice_obj[:]
            
            self.assertIn("slicing", str(context.exception).lower())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_gds_different_dtypes(self):
        """Test GDS with different data types."""
        # Create tensors with different dtypes
        dtypes_tensors = {
            "float32": torch.randn(50, 50, dtype=torch.float32),
            "float16": torch.randn(50, 50, dtype=torch.float16),
            "int32": torch.randint(0, 100, (50, 50), dtype=torch.int32),
        }
        
        dtype_file = tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors")
        dtype_file.close()
        
        try:
            save_file(dtypes_tensors, dtype_file.name)
            
            with safe_open(dtype_file.name, framework="pt", device="cuda:0", use_gds=True) as f:
                for name, expected_tensor in dtypes_tensors.items():
                    tensor = f.get_tensor(name)
                    self.assertEqual(tensor.dtype, expected_tensor.dtype)
                    self.assertEqual(tensor.device.type, "cuda")
        finally:
            if os.path.exists(dtype_file.name):
                os.unlink(dtype_file.name)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_gds_large_tensor(self):
        """Test GDS with a large tensor to verify performance benefit."""
        # Create a large tensor (100 MB)
        large_tensor = torch.randn(1000, 10000)
        large_file = tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors")
        large_file.close()
        
        try:
            save_file({"large": large_tensor}, large_file.name)
            
            # Load with GDS
            with safe_open(large_file.name, framework="pt", device="cuda:0", use_gds=True) as f:
                tensor_gds = f.get_tensor("large")
            
            # Verify correctness
            self.assertEqual(tensor_gds.shape, large_tensor.shape)
            self.assertEqual(tensor_gds.device.type, "cuda")
        finally:
            if os.path.exists(large_file.name):
                os.unlink(large_file.name)

    def test_gds_without_cuda(self):
        """Test that GDS without CUDA device fails gracefully."""
        if not torch.cuda.is_available():
            with self.assertRaises(ValueError):
                with safe_open(self.test_file.name, framework="pt", device="cpu", use_gds=True) as f:
                    pass


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)

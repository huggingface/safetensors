import sys
import unittest

import torch
from packaging.version import Version

from safetensors import safe_open
from safetensors.torch import load, load_file, save, save_file


try:
    import torch_npu  # noqa

    npu_present = True
except Exception:
    npu_present = False


class TorchTestCase(unittest.TestCase):
    def test_serialization(self):
        data = torch.zeros((2, 2), dtype=torch.int32)
        out = save({"test": data})

        self.assertEqual(
            out,
            b'@\x00\x00\x00\x00\x00\x00\x00{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}   '
            b" \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        )

        save_file({"test": data}, "serialization.safetensors")
        out = open("serialization.safetensors", "rb").read()
        self.assertEqual(
            out,
            b'@\x00\x00\x00\x00\x00\x00\x00{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}   '
            b" \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        )

        data[1, 1] = 1
        out = save({"test": data})

        self.assertEqual(
            out,
            b'@\x00\x00\x00\x00\x00\x00\x00{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}   '
            b" \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00",
        )
        save_file({"test": data}, "serialization.safetensors")
        out = open("serialization.safetensors", "rb").read()
        self.assertEqual(
            out,
            b'@\x00\x00\x00\x00\x00\x00\x00{"test":{"dtype":"I32","shape":[2,2],"data_offsets":[0,16]}}   '
            b" \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00",
        )

        data = torch.ones((2, 2), dtype=torch.bfloat16)
        data[0, 0] = 2.25
        out = save({"test": data})
        self.assertEqual(
            out,
            b'@\x00\x00\x00\x00\x00\x00\x00{"test":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]}}    \x10@\x80?\x80?\x80?',
        )

    def test_odd_dtype(self):
        data = {
            "test": torch.randn((2, 2), dtype=torch.bfloat16),
            "test2": torch.randn((2, 2), dtype=torch.float16),
            "test3": torch.zeros((2, 2), dtype=torch.bool),
            "test4": torch.zeros((2, 2), dtype=torch.complex64),
        }

        # Modify bool to have both values.
        data["test3"][0, 0] = True
        local = "./tests/data/out_safe_pt_mmap_small.safetensors"

        save_file(data, local)
        reloaded = load_file(local)
        self.assertTrue(torch.equal(data["test"], reloaded["test"]))
        self.assertTrue(torch.equal(data["test2"], reloaded["test2"]))
        self.assertTrue(torch.equal(data["test3"], reloaded["test3"]))
        self.assertTrue(torch.equal(data["test4"], reloaded["test4"]))

    def test_complex(self):
        # Test complex separately. Each value consists of two numbers
        # and we want to validate that the representation is the same
        # across platforms.
        data = torch.zeros((2, 2), dtype=torch.complex64)
        out = save({"test": data})

        self.assertEqual(
            out,
            b'@\x00\x00\x00\x00\x00\x00\x00{"test":{"dtype":"C64","shape":[2,2],"data_offsets":[0,32]}}    '
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        )

        real = torch.tensor([-1.0])
        imag = torch.tensor([1.0])
        data[1][1] = torch.complex(real, imag)
        out = save({"test": data})

        self.assertEqual(
            out,
            b'@\x00\x00\x00\x00\x00\x00\x00{"test":{"dtype":"C64","shape":[2,2],"data_offsets":[0,32]}}    '
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\xbf\x00\x00\x80?",
        )

    def test_odd_dtype_fp8(self):
        if torch.__version__ < "2.1":
            return  # torch.float8 requires 2.1

        data = {
            "test1": torch.tensor([-0.5], dtype=torch.float8_e4m3fn),
            "test2": torch.tensor([-0.5], dtype=torch.float8_e5m2),
        }
        local = "./tests/data/out_safe_pt_mmap_small.safetensors"

        save_file(data, local)
        reloaded = load_file(local)
        # note: PyTorch doesn't implement torch.equal for float8 so we just compare the single element
        self.assertEqual(reloaded["test1"].dtype, torch.float8_e4m3fn)
        self.assertEqual(reloaded["test1"].item(), -0.5)
        self.assertEqual(reloaded["test2"].dtype, torch.float8_e5m2)
        self.assertEqual(reloaded["test2"].item(), -0.5)

    def test_odd_dtype_fp4(self):
        if Version(torch.__version__) < Version("2.8"):
            return  # torch.float4 requires 2.8

        test1 = torch.tensor([0.0], dtype=torch.float8_e8m0fnu)
        test2 = torch.empty(2, 2, device="cpu", dtype=torch.float4_e2m1fn_x2)
        data = {
            "test1": test1,
            "test2": test2,
        }
        local = "./tests/data/out_safe_pt_mmap_fp4.safetensors"

        save_file(data, local)
        reloaded = load_file(local)
        # note: PyTorch doesn't implement torch.equal for float8 so we just compare the single element
        self.assertEqual(reloaded["test1"].dtype, torch.float8_e8m0fnu)
        self.assertEqual(reloaded["test1"], test1)
        self.assertEqual(reloaded["test2"].dtype, torch.float4_e2m1fn_x2)
        # TODO RuntimeError: "eq_cpu" not implemented for 'Float4_e2m1fn_x2'
        # self.assertEqual(reloaded["test2"], test2)

    def test_zero_sized(self):
        data = {
            "test": torch.zeros((2, 0), dtype=torch.float),
        }
        local = "./tests/data/out_safe_pt_mmap_small2.safetensors"
        save_file(data, local)
        reloaded = load_file(local)
        self.assertTrue(torch.equal(data["test"], reloaded["test"]))
        reloaded = load(open(local, "rb").read())
        self.assertTrue(torch.equal(data["test"], reloaded["test"]))

    def test_multiple_zero_sized(self):
        data = {
            "test": torch.zeros((2, 0), dtype=torch.float),
            "test2": torch.zeros((2, 0), dtype=torch.float),
        }
        local = "./tests/data/out_safe_pt_mmap_small3.safetensors"
        save_file(data, local)
        reloaded = load_file(local)
        self.assertTrue(torch.equal(data["test"], reloaded["test"]))
        self.assertTrue(torch.equal(data["test2"], reloaded["test2"]))

    def test_disjoint_tensors_shared_storage(self):
        A = torch.zeros((10, 10))
        data = {
            "test": A[1:],
            "test2": A[:1],
        }
        local = "./tests/data/out_safe_pt_mmap_small4.safetensors"
        save_file(data, local)

    def test_meta_tensor(self):
        A = torch.zeros((10, 10), device=torch.device("meta"))
        data = {
            "test": A,
        }
        local = "./tests/data/out_safe_pt_mmap_small5.safetensors"
        with self.assertRaises(RuntimeError) as ex:
            save_file(data, local)
        self.assertIn("Cannot copy out of meta tensor", str(ex.exception))

    def test_in_memory(self):
        data = {
            "test": torch.zeros((2, 2), dtype=torch.float32),
        }
        binary = save(data)
        self.assertEqual(
            binary,
            # Spaces are for forcing the alignment.
            b'@\x00\x00\x00\x00\x00\x00\x00{"test":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}   '
            b" \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        )
        reloaded = load(binary)
        self.assertTrue(torch.equal(data["test"], reloaded["test"]))

    @unittest.skipIf(not torch.cuda.is_available(), "Cuda is not available")
    def test_gpu(self):
        data = {
            "test": torch.arange(4).view((2, 2)).to("cuda:0"),
        }
        local = "./tests/data/out_safe_pt_mmap_small.safetensors"
        save_file(data, local)
        reloaded = load_file(local)
        self.assertTrue(torch.equal(torch.arange(4).view((2, 2)), reloaded["test"]))

    @unittest.skipIf(not torch.cuda.is_available(), "Cuda is not available")
    def test_gpu_default_device(self):
        data = {
            "test": torch.arange(4).view((2, 2)).to("cuda:0"),
        }
        local = "./tests/data/out_safe_pt_mmap_small.safetensors"
        save_file(data, local)
        with torch.device("cuda:0"):
            reloaded = load_file(local)
            assert reloaded["test"].device == torch.device("cpu")
            reloaded = load_file(local, device="cuda:0")
            assert reloaded["test"].device == torch.device("cuda:0")
            self.assertTrue(torch.equal(torch.arange(4).view((2, 2)), reloaded["test"]))

    @unittest.skipIf(not npu_present, "Npu is not available")
    def test_npu(self):
        data = {
            "test1": torch.zeros((2, 2), dtype=torch.float32).to("npu:0"),
            "test2": torch.zeros((2, 2), dtype=torch.float16).to("npu:0"),
        }
        local = "./tests/data/out_safe_pt_mmap_small_npu.safetensors"
        save_file(data, local)

        reloaded = load_file(local, device="npu:0")
        for k, v in reloaded.items():
            self.assertTrue(torch.allclose(data[k], reloaded[k]))

    def test_hpu(self):
        # must be run to load torch with Intel Gaudi bindings
        try:
            import habana_frameworks.torch.core as htcore  # noqa: F401
        except ImportError:
            self.skipTest("HPU is not available")

        data = {
            "test1": torch.zeros((2, 2), dtype=torch.float32).to("hpu"),
            "test2": torch.zeros((2, 2), dtype=torch.float16).to("hpu"),
        }
        local = "./tests/data/out_safe_pt_mmap_small_hpu.safetensors"
        save_file(data, local)

        reloaded = load_file(local, device="hpu")
        for k, v in reloaded.items():
            self.assertTrue(torch.allclose(data[k], reloaded[k]))

    @unittest.skipIf(not torch.cuda.is_available(), "Cuda is not available")
    def test_anonymous_accelerator(self):
        data = {
            "test1": torch.zeros((2, 2), dtype=torch.float32).to(device=0),
            "test2": torch.zeros((2, 2), dtype=torch.float16).to(device=0),
        }
        local = "./tests/data/out_safe_pt_mmap_small_anonymous.safetensors"
        save_file(data, local)

        reloaded = load_file(local, device=0)
        for k, v in reloaded.items():
            self.assertTrue(torch.allclose(data[k], reloaded[k]))

    def test_sparse(self):
        data = {"test": torch.sparse_coo_tensor(size=(2, 3))}
        local = "./tests/data/out_safe_pt_sparse.safetensors"
        with self.assertRaises(ValueError) as ctx:
            save_file(data, local)
        self.assertEqual(
            str(ctx.exception),
            "You are trying to save a sparse tensors: `['test']` which this library does not support. You can make it"
            " a dense tensor before saving with `.to_dense()` but be aware this might make a much larger file than"
            " needed.",
        )

    def test_bogus(self):
        data = {"test": {"some": "thing"}}
        local = "./tests/data/out_safe_pt_sparse.safetensors"
        with self.assertRaises(ValueError) as ctx:
            save_file(data, local)
        self.assertEqual(
            str(ctx.exception),
            "Key `test` is invalid, expected torch.Tensor but received <class 'dict'>",
        )

        with self.assertRaises(ValueError) as ctx:
            save_file("notadict", local)
        self.assertEqual(
            str(ctx.exception),
            "Expected a dict of [str, torch.Tensor] but received <class 'str'>",
        )


class LoadTestCase(unittest.TestCase):
    def setUp(self):
        data = {
            "test": torch.zeros((1024, 1024), dtype=torch.float32),
            "test2": torch.zeros((1024, 1024), dtype=torch.float32),
            "test3": torch.zeros((1024, 1024), dtype=torch.float32),
        }
        self.pt_filename = "./tests/data/pt_load.pt"
        self.sf_filename = "./tests/data/pt_load.safetensors"

        with open(self.pt_filename, "wb") as f:
            torch.save(data, f)

        save_file(data, self.sf_filename)

    def test_deserialization_safe(self):
        tweights = torch.load(self.pt_filename)
        weights = load_file(self.sf_filename)

        for k, v in weights.items():
            tv = tweights[k]
            self.assertTrue(torch.allclose(v, tv))
            self.assertEqual(v.device, torch.device("cpu"))

    @unittest.skipIf(not torch.cuda.is_available(), "Cuda is not available")
    def test_deserialization_device(self):
        with torch.device("cuda:0"):
            weights = load_file(self.sf_filename)
            self.assertEqual(weights["test"].device, torch.device("cpu"))

        torch.set_default_device(torch.device("cuda:0"))
        weights = load_file(self.sf_filename)
        self.assertEqual(weights["test"].device, torch.device("cpu"))
        torch.set_default_device(torch.device("cpu"))

        torch.set_default_device(torch.device(0))
        weights = load_file(self.sf_filename)
        self.assertEqual(weights["test"].device, torch.device("cpu"))
        torch.set_default_device(torch.device("cpu"))

    @unittest.skipIf(not torch.cuda.is_available(), "Cuda is not available")
    def test_deserialization_safe_gpu(self):
        # First time to hit disk
        tweights = torch.load(self.pt_filename, map_location="cuda:0")

        load_file(self.sf_filename, device=0)
        weights = load_file(self.sf_filename, device="cuda:0")

        for k, v in weights.items():
            tv = tweights[k]
            self.assertTrue(torch.allclose(v, tv))
            self.assertEqual(v.device, torch.device("cuda:0"))

    @unittest.skipIf(not torch.cuda.is_available(), "Cuda is not available")
    def test_deserialization_safe_gpu_slice(self):
        weights = {}
        with safe_open(self.sf_filename, framework="pt", device="cuda:0") as f:
            for k in f.keys():
                weights[k] = f.get_slice(k)[:1]
        tweights = torch.load(self.pt_filename, map_location="cuda:0")
        tweights = {k: v[:1] for k, v in tweights.items()}
        for k, v in weights.items():
            tv = tweights[k]
            self.assertTrue(torch.allclose(v, tv))
            self.assertEqual(v.device, torch.device("cuda:0"))

    @unittest.skipIf(torch.cuda.device_count() < 2, "Only 1 device available")
    def test_deserialization_safe_device_1(self):
        # Some GPUs may be visible but unavailable (e.g., NVLink/fabric issues)
        try:
            torch.zeros(1, device="cuda:1")
        except Exception:
            self.skipTest("cuda:1 is not usable")
        load_file(self.sf_filename, device=1)
        weights = load_file(self.sf_filename, device="cuda:1")
        tweights = torch.load(self.pt_filename, map_location="cuda:1")
        for k, v in weights.items():
            tv = tweights[k]
            self.assertTrue(torch.allclose(v, tv))
            self.assertEqual(v.device, torch.device("cuda:1"))

    @unittest.skipIf(not torch.cuda.is_available(), "Cuda is not available")
    def test_device_map_cpu_cuda(self):
        """Test device_map with mixed CPU and CUDA devices."""
        device_map = {
            "test": "cpu",
            "test2": "cuda:0",
            "test3": "cpu",
            "": "cpu",
        }
        with safe_open(self.sf_filename, framework="pt", device=device_map) as f:
            t1 = f.get_tensor("test")
            t2 = f.get_tensor("test2")
            t3 = f.get_tensor("test3")
            self.assertEqual(t1.device, torch.device("cpu"))
            self.assertEqual(t2.device, torch.device("cuda:0"))
            self.assertEqual(t3.device, torch.device("cpu"))

    @unittest.skipIf(not torch.cuda.is_available(), "Cuda is not available")
    def test_device_map_iter_tensors(self):
        """Test device_map with iter_tensors."""
        device_map = {
            "test": "cpu",
            "test2": "cuda:0",
            "test3": "cpu",
            "": "cpu",
        }
        with safe_open(self.sf_filename, framework="pt", device=device_map) as f:
            tensors = dict(f.iter_tensors())
            self.assertEqual(tensors["test"].device, torch.device("cpu"))
            self.assertEqual(tensors["test2"].device, torch.device("cuda:0"))
            self.assertEqual(tensors["test3"].device, torch.device("cpu"))

    @unittest.skipIf(not torch.cuda.is_available(), "Cuda is not available")
    def test_device_map_single_cuda(self):
        """Test device_map where all tensors go to single CUDA device (optimized path)."""
        device_map = {
            "test": "cuda:0",
            "test2": "cuda:0",
            "test3": "cuda:0",
            "": "cuda:0",
        }
        with safe_open(self.sf_filename, framework="pt", device=device_map) as f:
            t1 = f.get_tensor("test")
            t2 = f.get_tensor("test2")
            t3 = f.get_tensor("test3")
            self.assertEqual(t1.device, torch.device("cuda:0"))
            self.assertEqual(t2.device, torch.device("cuda:0"))
            self.assertEqual(t3.device, torch.device("cuda:0"))


class SliceTestCase(unittest.TestCase):
    def setUp(self):
        self.tensor = torch.arange(6, dtype=torch.float32).reshape((1, 2, 3))
        self.data = {"test": self.tensor}
        self.local = "./tests/data/out_safe_pt_mmap_slice.safetensors"
        # Need to copy since that call mutates the tensors to numpy
        save_file(self.data.copy(), self.local)

    def test_cannot_serialize_a_non_contiguous_tensor(self):
        tensor = torch.arange(6, dtype=torch.float32).reshape((1, 2, 3))
        x = tensor[:, :, 1]
        data = {"test": x}
        self.assertFalse(
            x.is_contiguous(),
        )
        with self.assertRaises(ValueError):
            save_file(data, "./tests/data/out.safetensors")

    def test_cannot_serialize_shared(self):
        A = torch.arange(6, dtype=torch.float32).reshape((2, 3))
        B = A[:1]
        data = {"A": A, "B": B}
        with self.assertRaises(RuntimeError):
            save_file(data, "./tests/data/out.safetensors")

        B = A[1:]
        data = {"A": A, "B": B}
        with self.assertRaises(RuntimeError):
            save_file(data, "./tests/data/out.safetensors")

    def test_deserialization_slice(self):
        with safe_open(self.local, framework="pt") as f:
            _slice = f.get_slice("test")
            self.assertEqual(_slice.get_shape(), [1, 2, 3])
            self.assertEqual(_slice.get_dtype(), "F32")
            tensor = _slice[:, :, 1:2]

        self.assertTrue(torch.equal(tensor, torch.Tensor([[[1.0], [4.0]]])))
        self.assertTrue(torch.equal(tensor, self.tensor[:, :, 1:2]))

        buffer = tensor.numpy()
        if sys.byteorder == "big":
            buffer.byteswap(inplace=True)
        buffer = buffer.tobytes()
        self.assertEqual(
            buffer,
            b"\x00\x00\x80?\x00\x00\x80@",
        )

    def test_deserialization_metadata(self):
        with safe_open(self.local, framework="pt") as f:
            metadata = f.metadata()
        self.assertEqual(metadata, None)

        # Save another one *with* metadata
        tensor = torch.arange(6, dtype=torch.float32).reshape((1, 2, 3))
        data = {"test": tensor}
        local = "./tests/data/out_safe_pt_mmap2.safetensors"
        # Need to copy since that call mutates the tensors to numpy
        save_file(data, local, metadata={"Something": "more"})
        with safe_open(local, framework="pt") as f:
            metadata = f.metadata()
        self.assertEqual(metadata, {"Something": "more"})


class DeviceMapTestCase(unittest.TestCase):
    """Tests for the device_map dict format passed to safe_open().

    Our device_map uses **exact tensor names** as keys, mapping each tensor
    to a target device. This differs from transformers' device_map which uses
    **module prefixes** (e.g., "model.layers.0" → 0) and from transformers'
    tp_plan which uses **wildcard patterns** (e.g., "layers.*.self_attn.q_proj"
    → "colwise").

    The special key "" sets the fallback device for unmapped tensors.
    """

    def setUp(self):
        # Create tensors with hierarchical names mimicking a real model
        self.tensors = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(64, 64),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(64, 64),
            "model.layers.1.self_attn.q_proj.weight": torch.randn(64, 64),
            "model.layers.1.self_attn.k_proj.weight": torch.randn(64, 64),
            "model.layers.1.mlp.gate_proj.weight": torch.randn(64, 64),
            "model.embed_tokens.weight": torch.randn(100, 64),
            "lm_head.weight": torch.randn(100, 64),
        }
        self.filename = "./tests/data/device_map_test.safetensors"
        save_file(self.tensors, self.filename)

    def test_device_map_exact_names_cpu(self):
        """Exact tensor name mapping works on CPU."""
        device_map = {
            "model.layers.0.self_attn.q_proj.weight": "cpu",
            "model.layers.0.self_attn.k_proj.weight": "cpu",
            "model.layers.0.mlp.gate_proj.weight": "cpu",
            "model.layers.1.self_attn.q_proj.weight": "cpu",
            "model.layers.1.self_attn.k_proj.weight": "cpu",
            "model.layers.1.mlp.gate_proj.weight": "cpu",
            "model.embed_tokens.weight": "cpu",
            "lm_head.weight": "cpu",
        }
        with safe_open(self.filename, framework="pt", device=device_map) as f:
            for name in self.tensors:
                t = f.get_tensor(name)
                self.assertEqual(t.device, torch.device("cpu"))
                self.assertTrue(torch.allclose(t, self.tensors[name]))

    def test_device_map_default_key(self):
        """The empty-string key provides a fallback for unmapped tensors."""
        # Only map one tensor explicitly, rest go to ""
        device_map = {
            "model.embed_tokens.weight": "cpu",
            "": "cpu",
        }
        with safe_open(self.filename, framework="pt", device=device_map) as f:
            for name in self.tensors:
                t = f.get_tensor(name)
                self.assertEqual(t.device, torch.device("cpu"))
                self.assertTrue(torch.allclose(t, self.tensors[name]))

    def test_device_map_default_is_cpu_when_omitted(self):
        """When "" is not specified, unmapped tensors go to CPU."""
        device_map = {
            "model.embed_tokens.weight": "cpu",
            # no "" key — should default to CPU
        }
        with safe_open(self.filename, framework="pt", device=device_map) as f:
            for name in self.tensors:
                t = f.get_tensor(name)
                self.assertEqual(t.device, torch.device("cpu"))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_device_map_mixed_cpu_cuda(self):
        """Mixed CPU/CUDA placement with exact tensor names."""
        device_map = {
            "model.layers.0.self_attn.q_proj.weight": "cuda:0",
            "model.layers.0.self_attn.k_proj.weight": "cuda:0",
            "model.layers.0.mlp.gate_proj.weight": "cuda:0",
            "model.embed_tokens.weight": "cuda:0",
            "": "cpu",  # layers.1.* and lm_head go to CPU
        }
        with safe_open(self.filename, framework="pt", device=device_map) as f:
            # Explicitly mapped to cuda:0
            t = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
            self.assertEqual(t.device, torch.device("cuda:0"))
            # Falls through to "" (cpu)
            t = f.get_tensor("model.layers.1.self_attn.q_proj.weight")
            self.assertEqual(t.device, torch.device("cpu"))
            t = f.get_tensor("lm_head.weight")
            self.assertEqual(t.device, torch.device("cpu"))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_device_map_values_correctness(self):
        """Tensor values are preserved through device_map loading."""
        device_map = {
            "model.layers.0.self_attn.q_proj.weight": "cuda:0",
            "": "cpu",
        }
        with safe_open(self.filename, framework="pt", device=device_map) as f:
            gpu_tensor = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
            cpu_tensor = f.get_tensor("model.layers.1.self_attn.q_proj.weight")
            self.assertTrue(torch.allclose(
                gpu_tensor.cpu(), self.tensors["model.layers.0.self_attn.q_proj.weight"]
            ))
            self.assertTrue(torch.allclose(
                cpu_tensor, self.tensors["model.layers.1.self_attn.q_proj.weight"]
            ))

    def test_device_map_prefix_matching(self):
        """Device map supports module-prefix matching (transformers-style).

        When an exact tensor name match is not found, the resolver walks up
        the module tree. So "model.layers.0" matches all tensors under that
        module, like "model.layers.0.self_attn.q_proj.weight".
        """
        prefix_map = {
            "model.layers.0": "cpu",
            "model.layers.1": "cpu",
            "model.embed_tokens": "cpu",
            "lm_head": "cpu",
            "": "cpu",
        }
        with safe_open(self.filename, framework="pt", device=prefix_map) as f:
            for name in self.tensors:
                t = f.get_tensor(name)
                self.assertEqual(t.device, torch.device("cpu"))
                self.assertTrue(torch.allclose(t, self.tensors[name]))

    def test_device_map_accepts_string_and_integer_values(self):
        """Dict device values accept both strings and integers.

        Both `device="cuda:0"` and `device=0` work in dict values.
        Integer values are treated as CUDA device indices.
        """
        # String device values work
        device_map = {
            "model.embed_tokens.weight": "cpu",
            "": "cpu",
        }
        with safe_open(self.filename, framework="pt", device=device_map) as f:
            t = f.get_tensor("model.embed_tokens.weight")
            self.assertEqual(t.device, torch.device("cpu"))

    @unittest.skipIf(torch.cuda.device_count() < 2, "Need 2+ GPUs")
    def test_device_map_multi_gpu_scatter(self):
        """Multi-GPU scatter: tensors land on correct devices.

        When 2+ CUDA devices appear in the device_map, the scatter subsystem
        reads each file once and distributes byte ranges to target GPUs.
        """
        try:
            torch.zeros(1, device="cuda:1")
        except Exception:
            self.skipTest("cuda:1 is not usable")

        device_map = {
            "model.layers.0.self_attn.q_proj.weight": "cuda:0",
            "model.layers.0.self_attn.k_proj.weight": "cuda:0",
            "model.layers.0.mlp.gate_proj.weight": "cuda:0",
            "model.embed_tokens.weight": "cuda:0",
            "model.layers.1.self_attn.q_proj.weight": "cuda:1",
            "model.layers.1.self_attn.k_proj.weight": "cuda:1",
            "model.layers.1.mlp.gate_proj.weight": "cuda:1",
            "lm_head.weight": "cuda:1",
        }
        with safe_open(self.filename, framework="pt", device=device_map) as f:
            for name, expected_device in device_map.items():
                if name == "":
                    continue
                t = f.get_tensor(name)
                self.assertEqual(t.device, torch.device(expected_device))
                self.assertTrue(torch.allclose(t.cpu(), self.tensors[name]))

    def test_transformers_device_map_expansion(self):
        """Document how to convert a transformers-style device_map to ours.

        Transformers uses module-prefix → device mappings:
            {"model.layers.0": 0, "model.layers.1": 1, "model.embed_tokens": 0}

        We need exact tensor names. This test shows the conversion pattern.
        """
        # Transformers-style prefix device_map
        hf_device_map = {
            "model.layers.0": "cpu",
            "model.layers.1": "cpu",
            "model.embed_tokens": "cpu",
            "lm_head": "cpu",
        }

        # Expand to per-tensor exact-name format
        tensor_names = list(self.tensors.keys())
        our_device_map = {}
        for tensor_name in tensor_names:
            parts = tensor_name.split(".")
            while parts:
                prefix = ".".join(parts)
                if prefix in hf_device_map:
                    our_device_map[tensor_name] = hf_device_map[prefix]
                    break
                parts.pop()

        # Every tensor should be resolved
        self.assertEqual(len(our_device_map), len(tensor_names))

        # Verify the expanded map works
        with safe_open(self.filename, framework="pt", device=our_device_map) as f:
            for name in tensor_names:
                t = f.get_tensor(name)
                self.assertEqual(t.device, torch.device("cpu"))
                self.assertTrue(torch.allclose(t, self.tensors[name]))

    def test_device_map_empty_string_default(self):
        """Empty string "" works as default key (transformers convention)."""
        device_map = {
            "model.embed_tokens": "cpu",
            "": "cpu",  # catch-all default
        }
        with safe_open(self.filename, framework="pt", device=device_map) as f:
            for name in self.tensors:
                t = f.get_tensor(name)
                self.assertEqual(t.device, torch.device("cpu"))
                self.assertTrue(torch.allclose(t, self.tensors[name]))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_device_map_integer_values(self):
        """Integer device values in dict are treated as CUDA devices."""
        device_map = {
            "": 0,  # integer → cuda:0
        }
        with safe_open(self.filename, framework="pt", device=device_map) as f:
            for name in self.tensors:
                t = f.get_tensor(name)
                self.assertEqual(t.device, torch.device("cuda:0"))
                self.assertTrue(torch.allclose(t.cpu(), self.tensors[name]))


class TpTestCase(unittest.TestCase):
    """Tests for tensor parallelism support in safe_open.

    Our TP implementation accepts a transformers-style tp_plan dict mapping
    wildcard patterns to slicing strategies ("colwise", "rowwise"), plus
    tp_rank and tp_world_size parameters. Tensors matching the plan are
    automatically sliced during I/O — colwise tensors get contiguous byte
    ranges (single NVMe read), rowwise tensors are loaded fully then narrowed.
    """

    def setUp(self):
        # Create tensors with known values so we can verify exact slices
        self.tensors = {
            "model.layers.0.self_attn.q_proj.weight": torch.arange(
                64 * 32, dtype=torch.float32
            ).reshape(64, 32),
            "model.layers.0.self_attn.q_proj.bias": torch.arange(
                64, dtype=torch.float32
            ),
            "model.layers.0.self_attn.o_proj.weight": torch.arange(
                32 * 64, dtype=torch.float32
            ).reshape(32, 64),
            "model.layers.0.self_attn.o_proj.bias": torch.arange(
                32, dtype=torch.float32
            ),
            "model.embed_tokens.weight": torch.arange(
                100 * 32, dtype=torch.float32
            ).reshape(100, 32),
        }
        self.filename = "./tests/data/tp_test.safetensors"
        save_file(self.tensors, self.filename)
        self.tp_plan = {
            "model.layers.*.self_attn.q_proj": "colwise",
            "model.layers.*.self_attn.o_proj": "rowwise",
        }

    def test_tp_colwise_2d_rank0(self):
        """Colwise slices dim -2: rank 0 gets first half of rows."""
        with safe_open(
            self.filename, framework="pt",
            tp_plan=self.tp_plan, tp_rank=0, tp_world_size=2,
        ) as f:
            t = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
            expected = self.tensors["model.layers.0.self_attn.q_proj.weight"][:32, :]
            self.assertEqual(t.shape, (32, 32))
            self.assertTrue(torch.allclose(t, expected))

    def test_tp_colwise_2d_rank1(self):
        """Colwise slices dim -2: rank 1 gets second half of rows."""
        with safe_open(
            self.filename, framework="pt",
            tp_plan=self.tp_plan, tp_rank=1, tp_world_size=2,
        ) as f:
            t = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
            expected = self.tensors["model.layers.0.self_attn.q_proj.weight"][32:64, :]
            self.assertEqual(t.shape, (32, 32))
            self.assertTrue(torch.allclose(t, expected))

    def test_tp_colwise_1d_rank0(self):
        """Colwise on 1D (bias) slices dim -1: rank 0 gets first half."""
        with safe_open(
            self.filename, framework="pt",
            tp_plan=self.tp_plan, tp_rank=0, tp_world_size=2,
        ) as f:
            t = f.get_tensor("model.layers.0.self_attn.q_proj.bias")
            expected = self.tensors["model.layers.0.self_attn.q_proj.bias"][:32]
            self.assertEqual(t.shape, (32,))
            self.assertTrue(torch.allclose(t, expected))

    def test_tp_colwise_1d_rank1(self):
        """Colwise on 1D (bias) slices dim -1: rank 1 gets second half."""
        with safe_open(
            self.filename, framework="pt",
            tp_plan=self.tp_plan, tp_rank=1, tp_world_size=2,
        ) as f:
            t = f.get_tensor("model.layers.0.self_attn.q_proj.bias")
            expected = self.tensors["model.layers.0.self_attn.q_proj.bias"][32:64]
            self.assertEqual(t.shape, (32,))
            self.assertTrue(torch.allclose(t, expected))

    def test_tp_rowwise_2d_rank0(self):
        """Rowwise slices dim -1: rank 0 gets first half of columns."""
        with safe_open(
            self.filename, framework="pt",
            tp_plan=self.tp_plan, tp_rank=0, tp_world_size=2,
        ) as f:
            t = f.get_tensor("model.layers.0.self_attn.o_proj.weight")
            expected = self.tensors["model.layers.0.self_attn.o_proj.weight"][:, :32]
            self.assertEqual(t.shape, (32, 32))
            self.assertTrue(torch.allclose(t, expected))

    def test_tp_rowwise_2d_rank1(self):
        """Rowwise slices dim -1: rank 1 gets second half of columns."""
        with safe_open(
            self.filename, framework="pt",
            tp_plan=self.tp_plan, tp_rank=1, tp_world_size=2,
        ) as f:
            t = f.get_tensor("model.layers.0.self_attn.o_proj.weight")
            expected = self.tensors["model.layers.0.self_attn.o_proj.weight"][:, 32:64]
            self.assertEqual(t.shape, (32, 32))
            self.assertTrue(torch.allclose(t, expected))

    def test_tp_rowwise_1d_replicated(self):
        """Rowwise on 1D (bias) is replicated — same for all ranks."""
        for rank in [0, 1]:
            with safe_open(
                self.filename, framework="pt",
                tp_plan=self.tp_plan, tp_rank=rank, tp_world_size=2,
            ) as f:
                t = f.get_tensor("model.layers.0.self_attn.o_proj.bias")
                expected = self.tensors["model.layers.0.self_attn.o_proj.bias"]
                self.assertEqual(t.shape, expected.shape)
                self.assertTrue(torch.allclose(t, expected))

    def test_tp_passthrough(self):
        """Tensors not in tp_plan are returned unchanged for all ranks."""
        for rank in [0, 1]:
            with safe_open(
                self.filename, framework="pt",
                tp_plan=self.tp_plan, tp_rank=rank, tp_world_size=2,
            ) as f:
                t = f.get_tensor("model.embed_tokens.weight")
                expected = self.tensors["model.embed_tokens.weight"]
                self.assertEqual(t.shape, expected.shape)
                self.assertTrue(torch.allclose(t, expected))

    def test_tp_uneven_split(self):
        """Uneven tensor dimension: ceil division distributes remainder."""
        tensors = {"w": torch.arange(65 * 32, dtype=torch.float32).reshape(65, 32)}
        filename = "./tests/data/tp_uneven.safetensors"
        save_file(tensors, filename)
        tp_plan = {"w": "colwise"}

        # Rank 0 gets ceil(65/2) = 33 rows
        with safe_open(
            filename, framework="pt",
            tp_plan=tp_plan, tp_rank=0, tp_world_size=2,
        ) as f:
            t = f.get_tensor("w")
            self.assertEqual(t.shape, (33, 32))
            self.assertTrue(torch.allclose(t, tensors["w"][:33, :]))

        # Rank 1 gets 65 - 33 = 32 rows
        with safe_open(
            filename, framework="pt",
            tp_plan=tp_plan, tp_rank=1, tp_world_size=2,
        ) as f:
            t = f.get_tensor("w")
            self.assertEqual(t.shape, (32, 32))
            self.assertTrue(torch.allclose(t, tensors["w"][33:65, :]))

    def test_tp_validation_errors(self):
        """TP parameter validation."""
        # tp_plan without tp_rank
        with self.assertRaises(Exception):
            safe_open(
                self.filename, framework="pt",
                tp_plan=self.tp_plan, tp_world_size=2,
            )

        # tp_plan without tp_world_size
        with self.assertRaises(Exception):
            safe_open(
                self.filename, framework="pt",
                tp_plan=self.tp_plan, tp_rank=0,
            )

        # tp_rank >= tp_world_size
        with self.assertRaises(Exception):
            safe_open(
                self.filename, framework="pt",
                tp_plan=self.tp_plan, tp_rank=2, tp_world_size=2,
            )

        # tp_world_size < 2
        with self.assertRaises(Exception):
            safe_open(
                self.filename, framework="pt",
                tp_plan=self.tp_plan, tp_rank=0, tp_world_size=1,
            )

        # Invalid strategy string
        with self.assertRaises(Exception):
            safe_open(
                self.filename, framework="pt",
                tp_plan={"w": "invalid_strategy"}, tp_rank=0, tp_world_size=2,
            )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_tp_colwise_gpu(self):
        """Colwise TP slice on GPU preserves values."""
        with safe_open(
            self.filename, framework="pt", device="cuda:0",
            tp_plan=self.tp_plan, tp_rank=0, tp_world_size=2,
        ) as f:
            t = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
            self.assertEqual(t.device, torch.device("cuda:0"))
            self.assertEqual(t.shape, (32, 32))
            expected = self.tensors["model.layers.0.self_attn.q_proj.weight"][:32, :]
            self.assertTrue(torch.allclose(t.cpu(), expected))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_tp_rowwise_gpu(self):
        """Rowwise TP narrow on GPU produces correct contiguous tensor."""
        with safe_open(
            self.filename, framework="pt", device="cuda:0",
            tp_plan=self.tp_plan, tp_rank=0, tp_world_size=2,
        ) as f:
            t = f.get_tensor("model.layers.0.self_attn.o_proj.weight")
            self.assertEqual(t.device, torch.device("cuda:0"))
            self.assertEqual(t.shape, (32, 32))
            self.assertTrue(t.is_contiguous())
            expected = self.tensors["model.layers.0.self_attn.o_proj.weight"][:, :32]
            self.assertTrue(torch.allclose(t.cpu(), expected))

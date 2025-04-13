import platform
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from safetensors import safe_open, serialize_file


class LoadTestCase(unittest.TestCase):
    """This test suite checks that :mod:`safetensors` Python extension tries to
    load :mod:`ml_dtypes` before resolving dtypes in case of NumPy, JAX, and
    TensorFlow (TF).
    """

    def setUp(self):
        # Emulate situation when `ml_dtypes` is not imported.
        sys.modules.pop("ml_dtypes", None)
        assert "ml_dtypes" not in sys.modules

        self.tmp_dir = TemporaryDirectory(prefix="safetensors-")
        self.path = Path(self.tmp_dir.name) / "ml_dtypes.safetensors"

        rng = np.random.default_rng(42)
        self.arr = rng.integers(low=0, high=np.iinfo(np.uint16).max, size=(2, 3), dtype=np.uint16)

        tensor_info = {
            "dtype": "bfloat16",
            "shape": [*self.arr.shape],
            "data": self.arr.tobytes(),
        }
        serialize_file({"arr": tensor_info}, self.path)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _test_deserialize(self, framework: str):
        st: safe_open
        with safe_open(self.path, framework) as st:
            arr = st.get_tensor("arr")
        assert "ml_dtypes" in sys.modules, "Package `ml_dtypes` has not been imported."
        import ml_dtypes

        assert arr.dtype == ml_dtypes.bfloat16

    def test_deserialize_numpy(self):
        self._test_deserialize("numpy")

    @unittest.skipIf(platform.system() == "Windows", "JAX is not available on Windows.")
    def test_deserialize_jax(self):
        self._test_deserialize("jax")

    def test_deserialize_tf(self):
        self._test_deserialize("tf")

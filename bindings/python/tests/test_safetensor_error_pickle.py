"""
Test that SafetensorError is picklable (fixes #684).
"""
import pickle
import unittest

from safetensors import SafetensorError


class SafetensorErrorPickleTestCase(unittest.TestCase):
    def test_safetensor_error_pickle_roundtrip_empty(self):
        err = SafetensorError()
        data = pickle.dumps(err)
        restored = pickle.loads(data)
        self.assertIsInstance(restored, SafetensorError)
        self.assertEqual(restored.args, err.args)

    def test_safetensor_error_pickle_roundtrip_with_message(self):
        err = SafetensorError("Error while deserializing header")
        data = pickle.dumps(err)
        restored = pickle.loads(data)
        self.assertIsInstance(restored, SafetensorError)
        self.assertEqual(restored.args, err.args)
        self.assertEqual(str(restored), str(err))

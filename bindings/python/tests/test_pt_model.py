import copy
import unittest

import torch

from safetensors.torch import (
    _find_shared_tensors,
    _is_complete,
    _remove_duplicate_names,
    load_model,
    save_file,
    save_model,
)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(100, 100)
        self.b = self.a


class CopyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(100, 100)
        self.b = copy.deepcopy(self.a)


class NoSharedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(100, 100)
        self.b = torch.nn.Linear(100, 100)


class TorchModelTestCase(unittest.TestCase):
    def test_is_complete(self):
        A = torch.zeros((3, 3))
        self.assertTrue(_is_complete(A))

        B = A[:1, :]
        self.assertFalse(_is_complete(B))

        # Covers the whole storage but with holes
        C = A[::2, :]
        self.assertFalse(_is_complete(C))

        D = torch.zeros((2, 2), device=torch.device("meta"))
        self.assertTrue(_is_complete(D))

    def test_find_shared_tensors(self):
        A = torch.zeros((3, 3))
        B = A[:1, :]

        self.assertEqual(_find_shared_tensors({"A": A, "B": B}), [{"A", "B"}])
        self.assertEqual(_find_shared_tensors({"A": A}), [{"A"}])
        self.assertEqual(_find_shared_tensors({"B": B}), [{"B"}])

        C = torch.zeros((2, 2), device=torch.device("meta"))
        D = C[:1]
        # Meta device is not shared
        self.assertEqual(_find_shared_tensors({"C": C, "D": D}), [])
        self.assertEqual(_find_shared_tensors({"C": C}), [])
        self.assertEqual(_find_shared_tensors({"D": D}), [])

    def test_remove_duplicate_names(self):
        A = torch.zeros((3, 3))
        B = A[:1, :]

        self.assertEqual(_remove_duplicate_names({"A": A, "B": B}), ["B"])
        self.assertEqual(_remove_duplicate_names({"A": A, "B": B, "C": A}), ["B", "C"])
        with self.assertRaises(RuntimeError):
            self.assertEqual(_remove_duplicate_names({"B": B}), [])

    def test_failure(self):
        model = Model()
        with self.assertRaises(RuntimeError):
            save_file(model.state_dict(), "tmp.safetensors")

    def test_workaround(self):
        model = Model()
        save_model(model, "tmp.safetensors")

        model2 = Model()
        load_model(model2, "tmp.safetensors")

        state_dict = model.state_dict()
        for k, v in model2.state_dict().items():
            torch.testing.assert_close(v, state_dict[k])

    def test_workaround_copy(self):
        model = CopyModel()
        save_model(model, "tmp.safetensors")

        model2 = CopyModel()
        load_model(model2, "tmp.safetensors")

        state_dict = model.state_dict()
        for k, v in model2.state_dict().items():
            torch.testing.assert_close(v, state_dict[k])

    def test_difference_with_torch(self):
        model = Model()
        torch.save(model.state_dict(), "tmp2.bin")

        model2 = NoSharedModel()
        # This passes on torch.
        # The tensors are shared on disk, they are *not* shared within the model
        # The model happily loads the tensors, and ends up *not* sharing the tensors by.
        # doing copies
        self.assertEqual(
            _find_shared_tensors(model2.state_dict()), [{"a.weight"}, {"a.bias"}, {"b.weight"}, {"b.bias"}]
        )
        model2.load_state_dict(torch.load("tmp2.bin"))
        self.assertEqual(
            _find_shared_tensors(model2.state_dict()), [{"a.weight"}, {"a.bias"}, {"b.weight"}, {"b.bias"}]
        )

        # However safetensors cannot save those, so we cannot
        # reload the saved file with the different model
        save_model(model, "tmp2.safetensors")
        with self.assertRaises(RuntimeError):
            load_model(model2, "tmp2.safetensors")

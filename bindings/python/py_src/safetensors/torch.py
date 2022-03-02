from safetensors import numpy
import numpy as np
from typing import Dict
import torch


def np2pt(numpy_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    for k, v in numpy_dict.items():
        numpy_dict[k] = torch.from_numpy(v)
    return numpy_dict


def pt2np(torch_dict: Dict[str, torch.Tensor]) -> Dict[str, np.array]:
    for k, v in torch_dict.items():
        torch_dict[k] = v.numpy()
    return torch_dict


def save(tensors: Dict[str, torch.Tensor]) -> bytes:
    np_tensors = pt2np(tensors)
    return numpy.save(np_tensors)


def save_file(tensors: Dict[str, torch.Tensor], filename: str):
    np_tensors = pt2np(tensors)
    return numpy.save_file(np_tensors, filename)


def load(buffer: bytes) -> Dict[str, torch.Tensor]:
    flat = numpy.load(buffer)
    return np2pt(flat)


def load_file(filename: str) -> Dict[str, torch.Tensor]:
    flat = numpy.load_file(filename)
    return np2pt(flat)

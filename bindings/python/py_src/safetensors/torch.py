from safetensors import numpy
import numpy as np
from typing import Dict
import torch


def np2pt(numpy_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    for k, v in numpy_dict.items():
        numpy_dict[k] = torch.from_numpy(v)
    return numpy_dict


def save(tensors: Dict[str, torch.Tensor]) -> bytes:
    for k, v in tensors.items():
        tensors[k] = v.numpy()
    return numpy.save(tensors)


def save_file(tensors: Dict[str, torch.Tensor], filename: str):
    for k, v in tensors.items():
        tensors[k] = v.numpy()
    return numpy.save_file(tensors, filename)


def load(buffer: bytes) -> Dict[str, torch.Tensor]:
    flat = numpy.load(buffer)
    return np2pt(flat)


def load_file(filename: str) -> Dict[str, torch.Tensor]:
    flat = numpy.load_file(filename)
    return np2pt(flat)

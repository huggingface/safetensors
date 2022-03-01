from safetensors.numpy_functions import save, save_file, load_file, load
import numpy as np
from typing import Dict
import torch


def np2pt(numpy_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    for k, v in numpy_dict.items():
        numpy_dict[k] = torch.from_numpy(v)
    return numpy_dict


def save_pt(tensors: Dict[str, torch.Tensor]) -> bytes:
    for k, v in tensors.items():
        tensors[k] = v.numpy()
    return save(tensors)


def save_file_pt(tensors: Dict[str, torch.Tensor], filename: str):
    for k, v in tensors.items():
        tensors[k] = v.numpy()
    return save_file(tensors, filename)


def load_pt(buffer: bytes) -> Dict[str, torch.Tensor]:
    flat = load(buffer)
    return np2pt(flat)


def load_file_pt(filename: str) -> Dict[str, torch.Tensor]:
    flat = load_file(filename)
    return np2pt(flat)

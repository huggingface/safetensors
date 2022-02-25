__version__ = "0.0.1"

from typing import Dict
import numpy as np
from .safetensors_rust import deserialize, serialize, deserialize_file
import torch


def save(tensor_dict: Dict[str, np.ndarray]) -> bytes:
    flattened = {
        k: {"dtype": v.dtype.name, "shape": v.shape, "data": v.tobytes()}
        for k, v in tensor_dict.items()
    }
    serialized = serialize(flattened)
    result = bytes(serialized)
    return result


TYPES = {"F32": np.float32, "I32": np.int32}


def getdtype(dtype_str: str) -> np.dtype:
    return TYPES[dtype_str]


def load_file(filename: str) -> Dict[str, np.ndarray]:
    flat = deserialize_file(filename)
    return to_numpy(flat)


def load(buffer: bytes) -> Dict[str, np.ndarray]:
    flat = deserialize(buffer)
    return to_numpy(flat)


def to_numpy(safeview):
    result = {}
    for k, v in safeview:
        dtype = getdtype(v["dtype"])
        arr = np.frombuffer(v["data"], dtype=dtype)
        result[k] = arr
    return result


def to_pt(numpy_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    for k, v in numpy_dict.items():
        numpy_dict[k] = torch.from_numpy(v)
    return numpy_dict


def save_pt(tensors: Dict[str, torch.Tensor]) -> bytes:
    for k, v in tensors.items():
        tensors[k] = v.numpy()
    return save(tensors)

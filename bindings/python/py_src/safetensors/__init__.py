__version__ = "0.0.1"

from typing import Dict
import numpy as np
from .safetensors_rust import deserialize, serialize
import torch


def save(tensor_dict: Dict[str, np.ndarray]) -> bytes:
    import datetime

    start = datetime.datetime.now()
    flattened = {
        k: {"dtype": v.dtype.name, "shape": v.shape, "data": v.tobytes()}
        for k, v in tensor_dict.items()
    }
    print("Flattened", datetime.datetime.now() - start)
    serialized = serialize(flattened)
    print("Serialized", datetime.datetime.now() - start)
    result = bytes(serialized)
    print("Bytes", datetime.datetime.now() - start)
    return result


TYPES = {"F32": np.float32, "I32": np.int32}


def getdtype(dtype_str: str) -> np.dtype:
    return TYPES[dtype_str]


def load(buffer: bytes) -> Dict[str, np.ndarray]:
    flat = deserialize(buffer)

    result = {}
    for k, v in flat:
        dtype = getdtype(v["dtype"])
        arr = np.frombuffer(v["data"], dtype=dtype)
        result[k] = arr
    return result


def load_pt(buffer: bytes) -> Dict[str, torch.Tensor]:
    out = load(buffer)
    for k, v in out.items():
        out[k] = torch.from_numpy(v)
    return out


def save_pt(tensors: Dict[str, torch.Tensor]) -> bytes:
    for k, v in tensors.items():
        tensors[k] = v.numpy()
    return save(tensors)

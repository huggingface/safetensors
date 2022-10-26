from importlib.metadata import metadata
from typing import Dict, Optional

import numpy as np

from .safetensors_rust import deserialize, safe_open, serialize, serialize_file


def save(tensor_dict: Dict[str, np.ndarray], metadata: Optional[Dict[str, str]] = None) -> bytes:
    flattened = {k: {"dtype": v.dtype.name, "shape": v.shape, "data": v.tobytes()} for k, v in tensor_dict.items()}
    serialized = serialize(flattened, metadata=metadata)
    result = bytes(serialized)
    return result


def load(filename: str) -> Dict[str, np.ndarray]:
    flat = deserialize(filename)
    return _view2np(flat)


def load_file(filename: str) -> Dict[str, np.ndarray]:
    result = {}
    with safe_open(filename, framework="np") as f:
        for k in f.keys():
            result[k] = f.get_tensor(k)
    return result


def save_file(tensor_dict: Dict[str, np.ndarray], filename: str, metadata: Optional[Dict[str, str]] = None):
    flattened = {k: {"dtype": v.dtype.name, "shape": v.shape, "data": v.tobytes()} for k, v in tensor_dict.items()}
    serialize_file(flattened, filename, metadata=metadata)


_TYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "U64": np.uint64,
    "I32": np.int32,
    "U32": np.uint32,
    "I16": np.int16,
    "U16": np.uint16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": bool,
}


def _getdtype(dtype_str: str) -> np.dtype:
    return _TYPES[dtype_str]


def _view2np(safeview) -> Dict[str, np.ndarray]:
    result = {}
    for k, v in safeview:
        dtype = _getdtype(v["dtype"])
        arr = np.frombuffer(v["data"], dtype=dtype).reshape(v["shape"])
        result[k] = arr
    return result

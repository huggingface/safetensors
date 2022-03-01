import numpy as np
from .safetensors_rust import deserialize, serialize, deserialize_file, serialize_file
from typing import Dict


def save(tensor_dict: Dict[str, np.ndarray]) -> bytes:
    flattened = {
        k: {"dtype": v.dtype.name, "shape": v.shape, "data": v.tobytes()}
        for k, v in tensor_dict.items()
    }
    serialized = serialize(flattened)
    result = bytes(serialized)
    return result


TYPES = {
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
    "BOOL": np.bool,
}


def getdtype(dtype_str: str) -> np.dtype:
    return TYPES[dtype_str]


def load_file(filename: str) -> Dict[str, np.ndarray]:
    flat = deserialize_file(filename)
    return view2np(flat)


def load(buffer: bytes) -> Dict[str, np.ndarray]:
    flat = deserialize(buffer)
    return view2np(flat)


def view2np(safeview):
    result = {}
    for k, v in safeview:
        dtype = getdtype(v["dtype"])
        arr = np.frombuffer(v["data"], dtype=dtype).reshape(v["shape"])
        result[k] = arr
    return result


def save_file(tensor_dict: Dict[str, np.ndarray], filename: str):
    flattened = {
        k: {"dtype": v.dtype.name, "shape": v.shape, "data": v.tobytes()}
        for k, v in tensor_dict.items()
    }
    serialize_file(flattened, filename)

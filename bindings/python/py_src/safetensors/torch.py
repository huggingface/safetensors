from .safetensors_rust import serialize_file, serialize, safe_open, deserialize
import math
from typing import Dict, Optional

import torch


def save(tensors: Dict[str, torch.Tensor], metadata: Optional[Dict[str, str]] = None) -> bytes:
    flattened = {
        k: {"dtype": str(v.dtype).split(".")[-1], "shape": v.shape, "data": _tobytes(k, v)}
        for k, v in tensors.items()
    }
    serialized = serialize(flattened, metadata=metadata)
    result = bytes(serialized)
    return result


def save_file(
    tensors: Dict[str, torch.Tensor],
    filename: str,
    metadata: Optional[Dict[str, str]] = None,
):
    flattened = {
        k: {"dtype": str(v.dtype).split(".")[-1], "shape": v.shape, "data": _tobytes(k, v)}
        for k, v in tensors.items()
    }
    serialize_file(flattened, filename, metadata=metadata)


def load_file(filename: str) -> Dict[str, torch.Tensor]:
    result = {}
    with safe_open(filename, framework="pt") as f:
        for k in f.keys():
            result[k] = f.get_tensor(k)
    return result


def load(filename: str) -> Dict[str, torch.Tensor]:
    flat = deserialize(filename)
    return _view2torch(flat)


_SIZE = {
    torch.int64: 8,
    torch.float32: 4,
    torch.int32: 4,
    torch.bfloat16: 2,
    torch.float16: 2,
    torch.int16: 2,
    torch.uint8: 1,
    torch.int8: 1,
    torch.bool: 1,
}

_TYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "I64": torch.int64,
    # "U64": torch.uint64,
    "I32": torch.int32,
    # "U32": torch.uint32,
    "I16": torch.int16,
    # "U16": torch.uint16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def _getdtype(dtype_str: str) -> torch.dtype:
    return _TYPES[dtype_str]


def _view2torch(safeview) -> Dict[str, torch.Tensor]:
    result = {}
    for k, v in safeview:
        dtype = _getdtype(v["dtype"])
        arr = torch.frombuffer(v["data"], dtype=dtype).reshape(v["shape"])
        result[k] = arr


def _tobytes(tensor: torch.Tensor, name: str) -> bytes:
    if not tensor.is_sparse():
        raise ValueError(
            f"""You are trying to save a sparse tensor: `{name}` which this library does not support. You can make it a dense
                tensor before saving with `.to_dense()` but be aware this might make a much larger file than needed."""
        )
    if not tensor.is_contiguous():
        raise ValueError(
            f"""You are trying to save a non contiguous tensor: `{name}` which is not allowed. It either means
        you are trying to save tensors which are reference of each other in which case it's recommended to save
        only the full tensors, and reslice at load time, or simply call `.contiguous()` on your tensor to pack it
        before saving."""
        )
    if tensor.device.type != "cpu":
        # Moving tensor to cpu before saving
        tensor = tensor.to("cpu")

    import ctypes
    import numpy as np

    length = np.prod(tensor.shape).item()
    bytes_per_item = _SIZE[tensor.dtype]

    total_bytes = length * bytes_per_item

    ptr = tensor.data_ptr()
    newptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte))

    data = np.ctypeslib.as_array(newptr, (total_bytes,))  # no internal copy

    return data.tobytes()

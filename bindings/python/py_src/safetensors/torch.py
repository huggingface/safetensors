import mmap
from typing import Any, Dict, Optional

import torch

from .safetensors_rust import deserialize, safe_open, serialize, serialize_file


def _flatten(tensors: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]:
    return {
        k: {
            "dtype": str(v.dtype).split(".")[-1],
            "shape": v.shape,
            "data": _tobytes(v, k),
        }
        for k, v in tensors.items()
    }


def save(tensors: Dict[str, torch.Tensor], metadata: Optional[Dict[str, str]] = None) -> bytes:
    serialized = serialize(_flatten(tensors), metadata=metadata)
    result = bytes(serialized)
    return result


def save_file(
    tensors: Dict[str, torch.Tensor],
    filename: str,
    metadata: Optional[Dict[str, str]] = None,
):
    serialize_file(_flatten(tensors), filename, metadata=metadata)


def load_file(filename: str) -> Dict[str, torch.Tensor]:
    result = {}
    with safe_open(filename, framework="pt") as f:
        with open(filename, mode="r+", encoding="utf8") as file_obj:
            with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_WRITE) as mmap_obj:
                for k in f.keys():
                    tensor_info = f.get_tensor_info(k)
                    data_offsets, shape, dtype_str = (
                        tensor_info["data_offsets"],
                        tensor_info["shape"],
                        tensor_info["dtype"],
                    )
                    idx_start, idx_end = data_offsets
                    dtype = _TYPES[dtype_str]
                    mmap_slice = mmap_obj[idx_start:idx_end]
                    result[k] = torch.frombuffer(mmap_slice, dtype=dtype).reshape(shape)
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

    try:
        if not tensor.is_contiguous():
            raise ValueError(
                f"You are trying to save a non contiguous tensor: `{name}` which is not allowed. It either means you"
                " are trying to save tensors which are reference of each other in which case it's recommended to save"
                " only the full tensors, and reslice at load time, or simply call `.contiguous()` on your tensor to"
                " pack it before saving."
            )
    except RuntimeError:
        # This occurs with sparse tensors
        raise ValueError(
            f"You are trying to save a sparse tensor: `{name}` which this library does not support."
            " You can make it a dense tensor before saving with `.to_dense()` but be aware this might"
            " make a much larger file than needed."
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

import os
import sys
from collections import defaultdict
from typing import Any, Dict, Optional, Union

import torch

from safetensors import deserialize, safe_open, serialize, serialize_file


def save(tensors: Dict[str, torch.Tensor], metadata: Optional[Dict[str, str]] = None) -> bytes:
    """
    Saves a dictionnary of tensors into raw bytes in safetensors format.

    Args:
        tensors (`Dict[str, torch.Tensor]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.

    Returns:
        `bytes`: The raw bytes representing the format

    Example:

    ```python
    from safetensors.torch import save
    import torch

    tensors = {"embedding": torch.zeros((512, 1024)), "attention": torch.zeros((256, 256))}
    byte_data = save(tensors)
    ```
    """
    serialized = serialize(_flatten(tensors), metadata=metadata)
    result = bytes(serialized)
    return result


def save_file(
    tensors: Dict[str, torch.Tensor],
    filename: Union[str, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
):
    """
    Saves a dictionnary of tensors into raw bytes in safetensors format.

    Args:
        tensors (`Dict[str, torch.Tensor]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        filename (`str`, or `os.PathLike`)):
            The filename we're saving into.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.

    Returns:
        `None`

    Example:

    ```python
    from safetensors.torch import save_file
    import torch

    tensors = {"embedding": torch.zeros((512, 1024)), "attention": torch.zeros((256, 256))}
    save(tensors, "model.safetensors")
    ```
    """
    serialize_file(_flatten(tensors), filename, metadata=metadata)


def load_file(filename: Union[str, os.PathLike], device="cpu") -> Dict[str, torch.Tensor]:
    """
    Loads a safetensors file into torch format.

    Args:
        filename (`str`, or `os.PathLike`)):
            The name of the file which contains the tensors
        device (`Dict[str, any]`, *optional*, defaults to `cpu`):
            The device where the tensors need to be located after load.
            available options are all regular torch device locations

    Returns:
        `Dict[str, torch.Tensor]`: dictionary that contains name as key, value as `torch.Tensor`

    Example:

    ```python
    from safetensors.torch import load_file

    file_path = "./my_folder/bert.safetensors"
    loaded = load_file(file_path)
    ```
    """
    result = {}
    with safe_open(filename, framework="pt", device=device) as f:
        for k in f.keys():
            result[k] = f.get_tensor(k)
    return result


def load(data: bytes) -> Dict[str, torch.Tensor]:
    """
    Loads a safetensors file into torch format from pure bytes.

    Args:
        data (`bytes`):
            The content of a safetensors file

    Returns:
        `Dict[str, torch.Tensor]`: dictionary that contains name as key, value as `torch.Tensor` on cpu

    Example:

    ```python
    from safetensors.torch import load

    file_path = "./my_folder/bert.safetensors"
    with open(file_path, "rb") as f:
        data = f.read()

    loaded = load(data)
    ```
    """
    flat = deserialize(data)
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
    torch.float64: 8,
}

_TYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
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

    return result


def _tobytes(tensor: torch.Tensor, name: str) -> bytes:
    if tensor.layout != torch.strided:
        raise ValueError(
            f"You are trying to save a sparse tensor: `{name}` which this library does not support."
            " You can make it a dense tensor before saving with `.to_dense()` but be aware this might"
            " make a much larger file than needed."
        )

    if not tensor.is_contiguous():
        raise ValueError(
            f"You are trying to save a non contiguous tensor: `{name}` which is not allowed. It either means you"
            " are trying to save tensors which are reference of each other in which case it's recommended to save"
            " only the full tensors, and reslice at load time, or simply call `.contiguous()` on your tensor to"
            " pack it before saving."
        )
    if tensor.device.type != "cpu":
        # Moving tensor to cpu before saving
        tensor = tensor.to("cpu")

    import ctypes

    import numpy as np

    # When shape is empty (scalar), np.prod returns a float
    # we need a int for the following calculations
    length = int(np.prod(tensor.shape).item())
    bytes_per_item = _SIZE[tensor.dtype]

    total_bytes = length * bytes_per_item

    ptr = tensor.data_ptr()
    newptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte))

    data = np.ctypeslib.as_array(newptr, (total_bytes,))  # no internal copy

    return data.tobytes()


def _flatten(tensors: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]:
    if sys.byteorder == "big":
        raise ValueError("Big endian is not supported, serialization need to be in little endian")
    if not isinstance(tensors, dict):
        raise ValueError(f"Expected a dict of [str, torch.Tensor] but received {type(tensors)}")
    ptrs = defaultdict(set)
    for k, v in tensors.items():
        if not isinstance(v, torch.Tensor):
            raise ValueError(f"Key `{k}` is invalid, expected torch.Tensor but received {type(v)}")

        if v.layout == torch.strided:
            ptrs[v.data_ptr()].add(k)

    failing = []
    for ptr, names in ptrs.items():
        if len(names) > 1:
            failing.append(names)

    if failing:
        raise RuntimeError(
            f"""Some tensors share memory, this will lead to duplicate memory on disk and potential differences when loading them again: {failing}"""
        )

    return {
        k: {
            "dtype": str(v.dtype).split(".")[-1],
            "shape": v.shape,
            "data": _tobytes(v, k),
        }
        for k, v in tensors.items()
    }

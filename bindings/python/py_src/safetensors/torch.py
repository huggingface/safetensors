from safetensors import numpy
from .safetensors_rust import serialize_file
import numpy as np
from typing import Dict
import torch


def save(tensors: Dict[str, torch.Tensor]) -> bytes:
    np_tensors = _pt2np(tensors)
    return numpy.save(np_tensors)


def save_file(tensors: Dict[str, torch.Tensor], filename: str):
    flattened = {
        k: {"dtype": str(v.dtype).split(".")[-1], "shape": v.shape, "data": _tobytes(v)}
        for k, v in tensors.items()
    }
    serialize_file(flattened, filename)


def load(buffer: bytes) -> Dict[str, torch.Tensor]:
    flat = numpy.load(buffer)
    return _np2pt(flat)


def load_file(filename: str) -> Dict[str, torch.Tensor]:
    flat = numpy.load_file(filename)
    return _np2pt(flat)


def _np2pt(numpy_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    for k, v in numpy_dict.items():
        numpy_dict[k] = torch.from_numpy(v)
    return numpy_dict


def _pt2np(torch_dict: Dict[str, torch.Tensor]) -> Dict[str, np.array]:
    for k, v in torch_dict.items():
        torch_dict[k] = v.numpy()
    return torch_dict


SIZE = {
    torch.bfloat16: 2,
    torch.float16: 2,
    torch.float32: 4,
    torch.uint8: 1,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 4,
    torch.int64: 8,
}


def _tobytes(tensor: torch.Tensor) -> bytes:
    import ctypes

    length = np.prod(tensor.shape)
    bytes_per_item = SIZE[tensor.dtype]

    total_bytes = length * bytes_per_item

    ptr = tensor.data_ptr()
    newptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte))

    data = np.ctypeslib.as_array(newptr, (total_bytes,))  # no internal copy

    return data.tobytes()

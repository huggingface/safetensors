from safetensors import numpy
from .safetensors_rust import serialize_file, read_metadata
import numpy as np
from typing import Dict, Optional
import torch


def np2pt(numpy_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    for k, v in numpy_dict.items():
        numpy_dict[k] = torch.from_numpy(v)
    return numpy_dict


def pt2np(torch_dict: Dict[str, torch.Tensor]) -> Dict[str, np.array]:
    for k, v in torch_dict.items():
        torch_dict[k] = v.numpy()
    return torch_dict


def save(tensors: Dict[str, torch.Tensor], metadata: Optional[Dict[str, str]] = None) -> bytes:
    np_tensors = pt2np(tensors)
    return numpy.save(np_tensors, metadata=metadata)


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


def tobytes(tensor: torch.Tensor) -> bytes:
    import ctypes

    length = np.prod(tensor.shape)
    bytes_per_item = SIZE[tensor.dtype]

    total_bytes = length * bytes_per_item

    ptr = tensor.data_ptr()
    newptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte))

    data = np.ctypeslib.as_array(newptr, (total_bytes,))  # no internal copy

    return data.tobytes()


def save_file(tensors: Dict[str, torch.Tensor], filename: str, metadata: Optional[Dict[str, str]] = None):
    flattened = {
        k: {"dtype": str(v.dtype).split(".")[-1], "shape": v.shape, "data": tobytes(v)}
        for k, v in tensors.items()
    }
    serialize_file(flattened, metadata, filename)


def load(buffer: bytes) -> Dict[str, torch.Tensor]:
    flat = numpy.load(buffer)
    return np2pt(flat)


def load_file(filename: str) -> Dict[str, torch.Tensor]:
    flat = numpy.load_file(filename)
    return np2pt(flat)


def read_metadata_in_file(filename: str) -> Dict[str, str]:
    return read_metadata(filename)

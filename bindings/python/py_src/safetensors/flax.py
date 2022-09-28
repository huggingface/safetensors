from safetensors import numpy
import numpy as np
from typing import Dict, Optional
import jax.numpy as jnp


def np2jnp(numpy_dict: Dict[str, np.ndarray]) -> Dict[str, jnp.DeviceArray]:
    for k, v in numpy_dict.items():
        numpy_dict[k] = jnp.array(v)
    return numpy_dict


def jnp2np(jnp_dict: Dict[str, jnp.DeviceArray]) -> Dict[str, np.array]:
    for k, v in jnp_dict.items():
        jnp_dict[k] = np.asarray(v)
    return jnp_dict


def save(tensors: Dict[str, jnp.DeviceArray], metadata: Optional[Dict[str, str]] = None) -> bytes:
    np_tensors = jnp2np(tensors)
    return numpy.save(np_tensors, metadata=metadata)


def save_file(tensors: Dict[str, jnp.DeviceArray], filename: str, metadata: Optional[Dict[str, str]] = None):
    np_tensors = jnp2np(tensors)
    return numpy.save_file(np_tensors, filename, metadata=metadata)


def load(buffer: bytes) -> Dict[str, jnp.DeviceArray]:
    flat = numpy.load(buffer)
    return np2jnp(flat)


def load_file(filename: str) -> Dict[str, jnp.DeviceArray]:
    flat = numpy.load_file(filename)
    return np2jnp(flat)


def read_metadata_in_file(filename: str) -> Dict[str, str]:
    return numpy.read_metadata(filename)
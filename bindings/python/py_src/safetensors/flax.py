from safetensors import numpy
import numpy as np
from typing import Dict
import jax.numpy as jnp


def np2jnp(numpy_dict: Dict[str, np.ndarray]) -> Dict[str, jnp.DeviceArray]:
    for k, v in numpy_dict.items():
        numpy_dict[k] = jnp.array(v)
    return numpy_dict


def jnp2np(jnp_dict: Dict[str, jnp.DeviceArray]) -> Dict[str, np.array]:
    for k, v in jnp_dict.items():
        jnp_dict[k] = np.asarray(v)
    return jnp_dict


def save(tensors: Dict[str, jnp.DeviceArray]) -> bytes:
    np_tensors = jnp2np(tensors)
    return numpy.save(np_tensors)


def save_file(tensors: Dict[str, jnp.DeviceArray], filename: str):
    np_tensors = jnp2np(tensors)
    return numpy.save_file(np_tensors, filename)


def load(buffer: bytes) -> Dict[str, jnp.DeviceArray]:
    flat = numpy.load(buffer)
    return np2jnp(flat)


def load_file(filename: str) -> Dict[str, jnp.DeviceArray]:
    flat = numpy.load_file(filename)
    return np2jnp(flat)

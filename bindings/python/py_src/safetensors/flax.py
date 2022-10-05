from typing import Dict, Optional

import numpy as np

import jax.numpy as jnp
from safetensors import numpy


def _np2jnp(numpy_dict: Dict[str, np.ndarray]) -> Dict[str, jnp.DeviceArray]:
    for k, v in numpy_dict.items():
        numpy_dict[k] = jnp.array(v)
    return numpy_dict


def _jnp2np(jnp_dict: Dict[str, jnp.DeviceArray]) -> Dict[str, np.array]:
    for k, v in jnp_dict.items():
        jnp_dict[k] = np.asarray(v)
    return jnp_dict


def save(tensors: Dict[str, jnp.DeviceArray], metadata: Optional[Dict[str, str]] = None) -> bytes:
    np_tensors = _jnp2np(tensors)
    return numpy.save(np_tensors)


def save_file(
    tensors: Dict[str, jnp.DeviceArray],
    filename: str,
    metadata: Optional[Dict[str, str]] = None,
):
    np_tensors = _jnp2np(tensors)
    return numpy.save_file(np_tensors, filename)


def load(buffer: bytes) -> Dict[str, jnp.DeviceArray]:
    flat = numpy.load(buffer)
    return _np2jnp(flat)


def load_file(filename: str) -> Dict[str, jnp.DeviceArray]:
    flat = numpy.load_file(filename)
    return _np2jnp(flat)

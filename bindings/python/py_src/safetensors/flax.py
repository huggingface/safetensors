from typing import Dict, Optional

import numpy as np

import jax.numpy as jnp
from safetensors import numpy


def save(tensors: Dict[str, jnp.DeviceArray], metadata: Optional[Dict[str, str]] = None) -> bytes:
    """
    Saves a dictionnary of tensors into raw bytes in safetensors format.

    Args:
        tensors (`Dict[str, jnp.DeviceArray]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.

    Returns:
        `bytes`: The raw bytes representing the format

    Example:

    ```python
    from safetensors.flax import save
    import flax

    tensors = {"embedding": jnp.zeros((512, 1024)), "attention": jnp.zeros((256, 256))}
    byte_data = save(tensors)
    ```
    """
    np_tensors = _jnp2np(tensors)
    return numpy.save(np_tensors, metadata=metadata)


def save_file(
    tensors: Dict[str, jnp.DeviceArray],
    filename: str,
    metadata: Optional[Dict[str, str]] = None,
):
    """
    Saves a dictionnary of tensors into raw bytes in safetensors format.

    Args:
        tensors (`Dict[str, jnp.DeviceArray]`):
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
    from safetensors.flax import save_file
    import flax

    tensors = {"embedding": jnp.zeros((512, 1024)), "attention": jnp.zeros((256, 256))}
    save(tensors, "model.safetensors")
    ```
    """
    np_tensors = _jnp2np(tensors)
    return numpy.save_file(np_tensors, filename, metadata=metadata)


def load(data: bytes) -> Dict[str, jnp.DeviceArray]:
    """
    Loads a safetensors file into flax format from pure bytes.

    Args:
        data (`bytes`):
            The content of a safetensors file

    Returns:
        `Dict[str, jnp.DeviceArray]`: dictionary that contains name as key, value as `jnp.DeviceArray` on cpu

    Example:

    ```python
    from safetensors.flax import load

    file_path = "./my_folder/bert.safetensors"
    with open(file_path, "rb") as f:
        data = f.read()

    loaded = load(data)
    ```
    """
    flat = numpy.load(data)
    return _np2jnp(flat)


def load_file(filename: str) -> Dict[str, jnp.DeviceArray]:
    """
    Loads a safetensors file into flax format.

    Args:
        filename (`str`, or `os.PathLike`)):
            The name of the file which contains the tensors
        device (`Dict[str, any]`, *optional*, defaults to `cpu`):
            The device where the tensors need to be located after load.
            available options are all regular flax device locations

    Returns:
        `Dict[str, jnp.DeviceArray]`: dictionary that contains name as key, value as `jnp.DeviceArray`

    Example:

    ```python
    from safetensors.flax import load_file

    file_path = "./my_folder/bert.safetensors"
    loaded = load_file(file_path)
    ```
    """
    flat = numpy.load_file(filename)
    return _np2jnp(flat)


def _np2jnp(numpy_dict: Dict[str, np.ndarray]) -> Dict[str, jnp.DeviceArray]:
    for k, v in numpy_dict.items():
        numpy_dict[k] = jnp.array(v)
    return numpy_dict


def _jnp2np(jnp_dict: Dict[str, jnp.DeviceArray]) -> Dict[str, np.array]:
    for k, v in jnp_dict.items():
        jnp_dict[k] = np.asarray(v)
    return jnp_dict

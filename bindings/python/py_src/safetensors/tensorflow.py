from typing import Dict, Optional

import numpy as np
import tensorflow as tf

from safetensors import numpy


def save(tensors: Dict[str, tf.Tensor], metadata: Optional[Dict[str, str]] = None) -> bytes:
    """
    Saves a dictionnary of tensors into raw bytes in safetensors format.

    Args:
        tensors (`Dict[str, tf.Tensor]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.

    Returns:
        `bytes`: The raw bytes representing the format

    Example:

    ```python
    from safetensors.tensorflow import save
    import tensorflow as tf

    tensors = {"embedding": tf.zeros((512, 1024)), "attention": tf.zeros((256, 256))}
    byte_data = save(tensors)
    ```
    """
    np_tensors = _tf2np(tensors)
    return numpy.save(np_tensors, metadata=metadata)


def save_file(
    tensors: Dict[str, tf.Tensor],
    filename: str,
    metadata: Optional[Dict[str, str]] = None,
):
    """
    Saves a dictionnary of tensors into raw bytes in safetensors format.

    Args:
        tensors (`Dict[str, tf.Tensor]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        filename (`str`):
            The filename we're saving into.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.

    Returns:
        `None`

    Example:

    ```python
    from safetensors.tensorflow import save_file
    import tensorflow as tf

    tensors = {"embedding": tf.zeros((512, 1024)), "attention": tf.zeros((256, 256))}
    save(tensors, "model.safetensors")
    ```
    """
    np_tensors = _tf2np(tensors)
    return numpy.save_file(np_tensors, filename, metadata=metadata)


def load(data: bytes) -> Dict[str, tf.Tensor]:
    """
    Loads a safetensors file into tensorflow format from pure bytes.

    Args:
        data (`bytes`):
            The content of a safetensors file

    Returns:
        `Dict[str, tf.Tensor]`: dictionary that contains name as key, value as `tf.Tensor` on cpu

    Example:

    ```python
    from safetensors.tensorflow import load

    file_path = "./my_folder/bert.safetensors"
    with open(file_path, "rb") as f:
        data = f.read()

    loaded = load(data)
    ```
    """
    flat = numpy.load(data)
    return _np2tf(flat)


def load_file(filename: str) -> Dict[str, tf.Tensor]:
    """
    Loads a safetensors file into tensorflow format.

    Args:
        filename (`str`):
            The name of the file which contains the tensors
        device (`Dict[str, any]`, *optional*, defaults to `cpu`):
            The device where the tensors need to be located after load.
            available options are all regular tensorflow device locations

    Returns:
        `Dict[str, tf.Tensor]`: dictionary that contains name as key, value as `tf.Tensor`

    Example:

    ```python
    from safetensors.tensorflow import load_file

    file_path = "./my_folder/bert.safetensors"
    loaded = load_file(file_path)
    ```
    """
    flat = numpy.load_file(filename)
    return _np2tf(flat)


def _np2tf(numpy_dict: Dict[str, np.ndarray]) -> Dict[str, tf.Tensor]:
    for k, v in numpy_dict.items():
        numpy_dict[k] = tf.convert_to_tensor(v)
    return numpy_dict


def _tf2np(tf_dict: Dict[str, tf.Tensor]) -> Dict[str, np.array]:
    for k, v in tf_dict.items():
        tf_dict[k] = v.numpy()
    return tf_dict

import os
from typing import Dict, Optional, Union

import numpy as np

import paddle
from safetensors import numpy


def save(tensors: Dict[str, paddle.Tensor], metadata: Optional[Dict[str, str]] = None) -> bytes:
    """
    Saves a dictionnary of tensors into raw bytes in safetensors format.

    Args:
        tensors (`Dict[str, paddle.Tensor]`):
            The incoming tensors. Tensors need to be contiguous and dense.
        metadata (`Dict[str, str]`, *optional*, defaults to `None`):
            Optional text only metadata you might want to save in your header.
            For instance it can be useful to specify more about the underlying
            tensors. This is purely informative and does not affect tensor loading.

    Returns:
        `bytes`: The raw bytes representing the format

    Example:

    ```python
    from safetensors.paddle import save
    import paddle

    tensors = {"embedding": paddle.zeros((512, 1024)), "attention": paddle.zeros((256, 256))}
    byte_data = save(tensors)
    ```
    """
    np_tensors = _paddle2np(tensors)
    return numpy.save(np_tensors, metadata=metadata)


def save_file(
    tensors: Dict[str, paddle.Tensor],
    filename: Union[str, os.PathLike],
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """
    Saves a dictionnary of tensors into raw bytes in safetensors format.

    Args:
        tensors (`Dict[str, paddle.Tensor]`):
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
    from safetensors.paddle import save_file
    import paddle

    tensors = {"embedding": paddle.zeros((512, 1024)), "attention": paddle.zeros((256, 256))}
    save_file(tensors, "model.safetensors")
    ```
    """
    np_tensors = _paddle2np(tensors)
    return numpy.save_file(np_tensors, filename, metadata=metadata)


def load(data: bytes, device: str = "cpu") -> Dict[str, paddle.Tensor]:
    """
    Loads a safetensors file into paddle format from pure bytes.

    Args:
        data (`bytes`):
            The content of a safetensors file

    Returns:
        `Dict[str, paddle.Tensor]`: dictionary that contains name as key, value as `paddle.Tensor` on cpu

    Example:

    ```python
    from safetensors.paddle import load

    file_path = "./my_folder/bert.safetensors"
    with open(file_path, "rb") as f:
        data = f.read()

    loaded = load(data)
    ```
    """
    flat = numpy.load(data)
    return _np2paddle(flat, device)


def load_file(filename: Union[str, os.PathLike], device="cpu") -> Dict[str, paddle.Tensor]:
    """
    Loads a safetensors file into paddle format.

    Args:
        filename (`str`, or `os.PathLike`)):
            The name of the file which contains the tensors
        device (`Dict[str, any]`, *optional*, defaults to `cpu`):
            The device where the tensors need to be located after load.
            available options are all regular paddle device locations

    Returns:
        `Dict[str, paddle.Tensor]`: dictionary that contains name as key, value as `paddle.Tensor`

    Example:

    ```python
    from safetensors.paddle import load_file

    file_path = "./my_folder/bert.safetensors"
    loaded = load_file(file_path)
    ```
    """
    flat = numpy.load_file(filename)
    output = _np2paddle(flat, device)
    return output


def _np2paddle(numpy_dict: Dict[str, np.ndarray], device: str = "cpu") -> Dict[str, paddle.Tensor]:
    for k, v in numpy_dict.items():
        numpy_dict[k] = paddle.to_tensor(v, place=device)
    return numpy_dict


def _paddle2np(paddle_dict: Dict[str, paddle.Tensor]) -> Dict[str, np.array]:
    for k, v in paddle_dict.items():
        paddle_dict[k] = v.detach().cpu().numpy()
    return paddle_dict

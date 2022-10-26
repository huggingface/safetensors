from typing import Dict, Optional

import numpy as np
import tensorflow as tf

from safetensors import numpy


def _np2tf(numpy_dict: Dict[str, np.ndarray]) -> Dict[str, tf.Tensor]:
    for k, v in numpy_dict.items():
        numpy_dict[k] = tf.convert_to_tensor(v)
    return numpy_dict


def _tf2np(tf_dict: Dict[str, tf.Tensor]) -> Dict[str, np.array]:
    for k, v in tf_dict.items():
        tf_dict[k] = v.numpy()
    return tf_dict


def save(tensors: Dict[str, tf.Tensor], metadata: Optional[Dict[str, str]] = None) -> bytes:
    np_tensors = _tf2np(tensors)
    return numpy.save(np_tensors, metadata=metadata)


def save_file(
    tensors: Dict[str, tf.Tensor],
    filename: str,
    metadata: Optional[Dict[str, str]] = None,
):
    np_tensors = _tf2np(tensors)
    return numpy.save_file(np_tensors, filename, metadata=metadata)


def load(buffer: bytes) -> Dict[str, tf.Tensor]:
    flat = numpy.load(buffer)
    return _np2tf(flat)


def load_file(filename: str) -> Dict[str, tf.Tensor]:
    flat = numpy.load_file(filename)
    return _np2tf(flat)

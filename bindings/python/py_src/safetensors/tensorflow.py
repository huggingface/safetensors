from safetensors import numpy
import numpy as np
from typing import Dict, Optional
import tensorflow as tf


def np2tf(numpy_dict: Dict[str, np.ndarray]) -> Dict[str, tf.DeviceArray]:
    for k, v in numpy_dict.items():
        numpy_dict[k] = tf.convert_to_tensor(v)
    return numpy_dict


def tf2np(tf_dict: Dict[str, tf.DeviceArray]) -> Dict[str, np.array]:
    for k, v in tf_dict.items():
        tf_dict[k] = v.numpy()
    return tf_dict


def save(tensors: Dict[str, tf.DeviceArray], metadata: Optional[Dict[str, str]] = None) -> bytes:
    np_tensors = tf2np(tensors)
    if metadata is None:
        metadata = {}
    if "format" not in metadata:
        metadata["format"] = "tf"
    return numpy.save(np_tensors, metadata=metadata)


def save_file(tensors: Dict[str, tf.DeviceArray], filename: str, metadata: Optional[Dict[str, str]] = None):
    np_tensors = tf2np(tensors)
    if metadata is None:
        metadata = {}
    if "format" not in metadata:
        metadata["format"] = "tf"
    return numpy.save_file(np_tensors, filename, metadata=metadata)


def load(buffer: bytes) -> Dict[str, tf.DeviceArray]:
    flat = numpy.load(buffer)
    return np2tf(flat)


def load_file(filename: str) -> Dict[str, tf.DeviceArray]:
    flat = numpy.load_file(filename)
    return np2tf(flat)

def read_metadata_in_file(filename: str) -> Dict[str, str]:
    return numpy.read_metadata(filename)
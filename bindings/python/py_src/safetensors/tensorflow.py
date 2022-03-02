from safetensors import numpy
import numpy as np
from typing import Dict
import tensorflow as tf


def np2tf(numpy_dict: Dict[str, np.ndarray]) -> Dict[str, tf.DeviceArray]:
    for k, v in numpy_dict.items():
        numpy_dict[k] = tf.convert_to_tensor(v)
    return numpy_dict


def tf2np(tf_dict: Dict[str, tf.DeviceArray]) -> Dict[str, np.array]:
    for k, v in tf_dict.items():
        tf_dict[k] = v.numpy()
    return tf_dict


def save(tensors: Dict[str, tf.DeviceArray]) -> bytes:
    np_tensors = tf2np(tensors)
    return numpy.save(np_tensors)


def save_file(tensors: Dict[str, tf.DeviceArray], filename: str):
    np_tensors = tf2np(tensors)
    return numpy.save_file(np_tensors, filename)


def load(buffer: bytes) -> Dict[str, tf.DeviceArray]:
    flat = numpy.load(buffer)
    return np2tf(flat)


def load_file(filename: str) -> Dict[str, tf.DeviceArray]:
    flat = numpy.load_file(filename)
    return np2tf(flat)

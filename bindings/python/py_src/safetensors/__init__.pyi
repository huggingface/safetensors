# Generated content DO NOT EDIT
@staticmethod
def deserialize(bytes):
    """
    Opens a safetensors lazily and returns tensors as asked

    Args:
        data (:obj:`bytes`):
            The byte content of a file

    Returns:
        (:obj:`List[str, Dict[str, Dict[str, any]]]`):
            The deserialized content is like:
                [("tensor_name", {"shape": [2, 3], "dtype": "F32", "data": b"\0\0.." }), (...)]
    """
    pass

@staticmethod
def serialize(tensor_dict, metadata=None):
    """
    Serializes raw data.

    Args:
        tensor_dict (:obj:`Dict[str, Dict[Any]]`):
            The tensor dict is like:
                {"tensor_name": {"dtype": "F32", "shape": [2, 3], "data": b"\0\0"}}
        metadata (:obj:`Dict[str, str]`, *optional*):
            The optional purely text annotations

    Returns:
        (:obj:`bytes`):
            The serialized content.
    """
    pass

@staticmethod
def serialize_file(tensor_dict, filename, metadata=None):
    """
    Serializes raw data.

    Args:
        tensor_dict (:obj:`Dict[str, Dict[Any]]`):
            The tensor dict is like:
                {"tensor_name": {"dtype": "F32", "shape": [2, 3], "data": b"\0\0"}}
        filename (:obj:`str`):
            The name of the file to write into.
        metadata (:obj:`Dict[str, str]`, *optional*):
            The optional purely text annotations

    Returns:
        (:obj:`bytes`):
            The serialized content.
    """
    pass

class safe_open:
    """
    Opens a safetensors lazily and returns tensors as asked

    Args:
        filename (:obj:`str`):
            The filename to open

        framework (:obj:`str`):
            The framework you want your tensors in. Supported values:
            `pt`, `tf`, `flax`, `numpy`.

        device (:obj:`str`, defaults to :obj:`"cpu"`):
            The device on which you want the tensors.
    """

    def __init__(self, filename, framework, device="cpu"):
        pass

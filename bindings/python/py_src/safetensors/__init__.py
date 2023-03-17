__version__ = "0.3.0"
import json
from typing import Dict

# Re-export this
from ._safetensors_rust import SafetensorError, deserialize, safe_open, serialize, serialize_file  # noqa: F401


def metadata(data: bytes) -> Dict[str, str]:
    """
    Loads the metadata of a safetensors byte representation

    Args:
        data (`bytes`):
            The content of a safetensors file

    Returns:
        `Dict[str, str]`: dictionary of the metadata in the file

    Example:

    ```python
    import safetensors

    file_path = "./my_folder/bert.safetensors"
    with open(file_path, "rb") as f:
        data = f.read()

    metadata = safetensors.metadata(data)
    ```
    """
    n_header = data[:8]
    n = int.from_bytes(n_header, "little")
    metadata_bytes = data[8 : 8 + n]
    header = json.loads(metadata_bytes)
    return header.get("__metadata__", {})

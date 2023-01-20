__version__ = "0.2.9"

# Re-export this
from ._safetensors_rust import safe_open, serialize, serialize_file, deserialize, SafetensorError  # noqa: F401

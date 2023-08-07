__version__ = "0.3.2.rc1"

# Re-export this
from ._safetensors_rust import SafetensorError, deserialize, safe_open, serialize, serialize_file  # noqa: F401

# Re-export this
from ._safetensors_rust import (  # noqa: F401
    SafetensorError,
    __version__,
    deserialize,
    deserialize_file_linux_io_uring,
    safe_open,
    _safe_open_handle,
    serialize,
    serialize_file,
)

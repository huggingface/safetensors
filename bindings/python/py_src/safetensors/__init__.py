# Re-export this
from ._safetensors_rust import (  # noqa: F401
    SafetensorError,
    __version__,
    deserialize,
    safe_open,
    _safe_open_handle,
    serialize,
    serialize_file,
    serialize_file_direct,
    serialize_file_direct_vectored,
    serialize_file_mmap,
    serialize_file_oneshot,
    serialize_file_parallel,
    serialize_file_vectored,
)

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

# Conditionally import safe_open_io_uring on Linux
import sys

if sys.platform == "linux":
    try:
        from ._safetensors_rust import safe_open_io_uring  # noqa: F401
    except ImportError:
        pass  # Not available on this platform

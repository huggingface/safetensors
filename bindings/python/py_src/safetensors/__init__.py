# Core functions available on all platforms
from ._safetensors_rust import (
    SafetensorError,
    __version__,
    deserialize,
    safe_open,
    _safe_open_handle,
    serialize,
    serialize_file,
)

# io_uring support - only available on Linux with supported architectures
# (x86_64, aarch64, riscv64, loongarch64, powerpc64, powerpc64le)
try:
    from ._safetensors_rust import (
        deserialize_file_io_uring,
        safe_open_io_uring,
    )
except ImportError:
    # io_uring not compiled for this platform
    pass

__all__ = [
    "SafetensorError",
    "__version__",
    "deserialize",
    "safe_open",
    "_safe_open_handle",
    "serialize",
    "serialize_file",
]

# Add io_uring exports if available
try:
    deserialize_file_io_uring
    __all__.extend(["deserialize_file_io_uring"])
except NameError:
    pass

try:
    safe_open_io_uring
    __all__.extend(["safe_open_io_uring"])
except NameError:
    pass

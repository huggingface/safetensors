# Re-export this
from ._safetensors_rust import (  # noqa: F401
    SafetensorError as _RustSafetensorError,
    __version__,
    deserialize,
    safe_open,
    _safe_open_handle,
    serialize,
    serialize_file,
)


class SafetensorError(_RustSafetensorError):
    """
    Custom Python Exception for Safetensor errors.
    Subclasses the Rust exception so it remains picklable (e.g. for multiprocessing).
    """

    def __reduce__(self):
        return (SafetensorError, self.args)

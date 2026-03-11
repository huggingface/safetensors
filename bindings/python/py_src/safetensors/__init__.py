# Re-export this
from ._safetensors_rust import (  # noqa: F401
    SafetensorError,
    __version__,
    deserialize,
    has_hmll,
    safe_open,
    _safe_open_handle,
    serialize,
    serialize_file,
)

if has_hmll:
    from ._safetensors_rust import safe_open_hmll

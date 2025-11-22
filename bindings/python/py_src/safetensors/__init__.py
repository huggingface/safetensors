# Re-export this
from ._safetensors_rust import (  # noqa: F401
    SafetensorError,
    __version__,
    deserialize,
    safe_open,
    _safe_open_handle,
    serialize,
    serialize_file,
)

# Export GDS support status for programmatic checking
from .torch import _is_gds_available  # noqa: F401

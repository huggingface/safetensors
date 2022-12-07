import datetime
import sys
import tempfile
from collections import defaultdict

import atheris


with atheris.instrument_imports():
    from safetensors.torch import load_file


EXCEPTIONS = defaultdict(int)
START = datetime.datetime.now()
DT = datetime.timedelta(seconds=30)


def TestOneInput(data):
    global START
    with tempfile.NamedTemporaryFile() as f:
        f.write(data)
        f.seek(0)
        try:
            load_file(f.name, device=0)
        except Exception as e:
            EXCEPTIONS[str(e)] += 1

    if datetime.datetime.now() - START > DT:
        for e, n in EXCEPTIONS.items():
            print(e, n)
        START = datetime.datetime.now()


atheris.Setup(sys.argv, TestOneInput)
atheris.Fuzz()

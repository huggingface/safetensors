import datetime
import sys
import tempfile
import time
from collections import Counter
import threading

import atheris

EXCEPTIONS = Counter()
START = time.time()
DT = 30
LOCK = threading.Lock()

def TestOneInput(data):
    global START
    global EXCEPTIONS
    with tempfile.NamedTemporaryFile(mode="wb") as f:
        f.write(data)
        f.seek(0)
        try:
            load_file(f.name, device=0)
        except Exception as e:
            with LOCK:
                EXCEPTIONS[str(e)] += 1

    if time.time() - START > DT:
        with LOCK:
            for e, n in EXCEPTIONS.items():
                print(e, n)
            START = time.time()


atheris.Setup(sys.argv, TestOneInput)
atheris.Fuzz()

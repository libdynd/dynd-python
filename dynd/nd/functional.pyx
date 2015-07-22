from .. import ndt
from .callable cimport callable, wrap_callable

def apply(tp_or_func, func = None):
    def make(tp, func):
        return wrap_array(_apply(func, tp))

    if func is None:
        if isinstance(tp_or_func, ndt.type):
            return lambda func: make(tp_or_func, func)

        return make(ndt.callable(tp_or_func), tp_or_func)

    return make(tp_or_func, func)

def elwise(func):
    if not isinstance(func, callable):
        func = apply(func)

    return wrap_callable(_elwise((<callable> func).v))
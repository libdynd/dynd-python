from .. import ndt
from .arrfunc cimport arrfunc, wrap_arrfunc

def apply(tp_or_func, func = None):
    def make(tp, func):
        return wrap_array(_apply(func, tp))

    if func is None:
        if isinstance(tp_or_func, ndt.type):
            return lambda func: make(tp_or_func, func)

        return make(ndt.arrfunc(tp_or_func), tp_or_func)

    return make(tp_or_func, func)

def elwise(func):
    if not isinstance(func, arrfunc):
        func = apply(func)

    return wrap_arrfunc(_elwise((<arrfunc> func).v))
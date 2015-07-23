from libcpp.vector cimport vector

from .. import ndt
from .callable cimport callable, _callable, wrap_callable

from dynd.wrapper cimport wrap, begin, end
from dynd.ndt.type cimport type, _type

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

def multidispatch(type tp, iterable = None, callable default = callable()):
    return wrap(_multidispatch(tp.v, begin[_callable](iterable),
        end[_callable](iterable), default.v))
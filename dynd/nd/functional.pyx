from dynd.wrapper cimport wrap, begin, end
from .. import ndt
from .callable cimport _callable, callable

from dynd.ndt.type cimport type

def apply(tp_or_func, func = None):
    def make(tp, func):
        return wrap(_apply(func, tp))

    if func is None:
        if isinstance(tp_or_func, ndt.type):
            return lambda func: make(tp_or_func, func)

        return make(ndt.callable(tp_or_func), tp_or_func)

    return make(tp_or_func, func)

def elwise(func):
    if not isinstance(func, callable):
        func = apply(func)

    return wrap(_elwise((<callable> func).v))

def multidispatch(type tp, iterable = None, callable default = callable()):
    return wrap(_multidispatch(tp.v, begin[_callable](iterable),
        end[_callable](iterable), default.v))
from ..cpp.func.callable cimport callable as _callable

cdef class callable(object):
    cdef _callable v
    cdef _callable as_cpp(callable self)
    @staticmethod
    cdef callable from_cpp(_callable v)

from ..cpp.func.callable cimport callable as _callable

cdef class callable(object):
    cdef _callable v

cdef _callable callable_to_cpp(callable) except *
cdef callable callable_from_cpp(_callable)

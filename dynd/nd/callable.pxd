from ..cpp.callable cimport callable as _callable

cdef class callable(object):
    cdef _callable v

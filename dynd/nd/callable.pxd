from ..config cimport translate_exception
from .array cimport array
from ..cpp.callable cimport callable as _callable

cdef class callable(object):
    cdef _callable v

from ..config cimport translate_exception
from .array cimport array
from .type cimport type

cdef extern from "dynd/view.hpp" namespace "dynd::nd" nogil:
    array view(const array &arr, const type &tp) except +translate_exception

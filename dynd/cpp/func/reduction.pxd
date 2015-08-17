from .callable cimport callable
from ...config cimport translate_exception

cdef extern from 'dynd/func/reduction.hpp' namespace 'dynd::nd::functional' nogil:
    callable reduction(callable) except +translate_exception

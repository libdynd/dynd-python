from .callable cimport callable
from ...config cimport translate_exception

cdef extern from 'dynd/func/elwise.hpp' namespace 'dynd::nd::functional' nogil:
    callable elwise(callable) except +translate_exception

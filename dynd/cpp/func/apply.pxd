from .callable cimport callable
from ...config cimport translate_exception

cdef extern from 'dynd/func/apply.hpp' namespace 'dynd::nd::functional' nogil:
    callable apply[T](T) except +translate_exception

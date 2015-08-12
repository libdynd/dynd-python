from .callable cimport callable

cdef extern from 'dynd/func/elwise.hpp' namespace 'dynd::nd::functional':
    callable elwise(callable) except +translate_exception

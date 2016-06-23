from ..config cimport translate_exception
from .callable cimport callable

cdef extern from "dynd/functional.hpp" namespace "dynd::nd::functional" nogil:
    callable apply(...) except +translate_exception
    callable elwise(callable) except +translate_exception
    callable reduction(callable, callable) except +translate_exception

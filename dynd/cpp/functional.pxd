from ..config cimport translate_exception
from .callable cimport callable

cdef extern from "dynd/functional.hpp" namespace "dynd::nd::functional" nogil:
    callable apply(...) except +translate_exception

from ..callable cimport callable

cdef extern from "dynd/arithmetic.hpp" namespace "dynd::nd" nogil:
    callable pow

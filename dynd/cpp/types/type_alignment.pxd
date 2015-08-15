from ..type cimport type

from ...config cimport translate_exception

cdef extern from "dynd/types/type_alignment.hpp" namespace "dynd::ndt":
    type make_unaligned(type&) except +translate_exception

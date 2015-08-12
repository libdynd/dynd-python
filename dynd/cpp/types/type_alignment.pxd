from ..type cimport type

from ...config cimport translate_exception

cdef extern from "dynd/types/type_alignment.hpp" namespace "dynd":
    type dynd_make_unaligned_type "dynd::ndt::make_unaligned" (type&) except +translate_exception

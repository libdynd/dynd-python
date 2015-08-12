from ..type cimport type

from ...config cimport translate_exception

cdef extern from "dynd/types/byteswap_type.hpp" namespace "dynd":
    type dynd_make_byteswap_type "dynd::ndt::byteswap_type::make" (type&) except +translate_exception
    type dynd_make_byteswap_type "dynd::ndt::byteswap_type::make" (type&, type&) except +translate_exception

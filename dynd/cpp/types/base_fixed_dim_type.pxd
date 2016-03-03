from libc.stdint cimport intptr_t

from ..type cimport type

from ...config cimport translate_exception

cdef extern from "dynd/types/base_fixed_dim_type.hpp" namespace "dynd":
    type dynd_make_fixed_dim_kind_type "dynd::ndt::base_fixed_dim_type::make" (type&) except +translate_exception
    type dynd_make_fixed_dim_kind_type "dynd::ndt::base_fixed_dim_type::make" (type&, intptr_t) except +translate_exception

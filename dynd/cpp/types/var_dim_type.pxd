from ..type cimport type

from ...config cimport translate_exception

cdef extern from "dynd/types/var_dim_type.hpp" namespace "dynd":
    type dynd_make_var_dim_type "dynd::ndt::var_dim_type::make" (type&) except +translate_exception

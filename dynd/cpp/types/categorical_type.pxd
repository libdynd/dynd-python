from ..type cimport type
from ..array cimport array

from ...config cimport translate_exception

cdef extern from "dynd/types/categorical_type.hpp" namespace "dynd::ndt":
    type dynd_make_categorical_type "dynd::ndt::categorical_type::make" (array&) except +translate_exception
    type factor_categorical(array&) except +translate_exception

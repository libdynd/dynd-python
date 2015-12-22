from ..type cimport type
from ...config cimport translate_exception

cdef extern from "dynd/types/datetime_type.hpp" namespace "dynd::ndt::datetime_type":
    type make() except +translate_exception

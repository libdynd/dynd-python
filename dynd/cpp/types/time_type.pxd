from ..type cimport type
from ...config cimport translate_exception

cdef extern from "dynd/types/time_type.hpp" namespace "dynd::ndt::time_type":
    type make() except +translate_exception

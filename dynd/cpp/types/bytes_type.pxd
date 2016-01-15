from ..type cimport type
from ...config cimport translate_exception

cdef extern from "dynd/types/bytes_type.hpp" namespace "dynd::ndt::bytes_type":
    type make() except +translate_exception

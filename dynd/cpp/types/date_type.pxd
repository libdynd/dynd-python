from ..type cimport type
from ...config cimport translate_exception

cdef extern from "dynd/types/date_type.hpp" namespace "dynd::ndt::date_type":
    type make() except +translate_exception

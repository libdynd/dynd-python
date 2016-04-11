from libcpp.string cimport string
from libcpp.vector cimport vector

from ..array cimport array
from ..type cimport type

from ...config cimport translate_exception

cdef extern from 'dynd/types/struct_type.hpp' namespace 'dynd::ndt':
    cppclass struct_type:
        pass

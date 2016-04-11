from libcpp.vector cimport vector

from ..type cimport type
from ..array cimport array

from ...config cimport translate_exception

cdef extern from 'dynd/types/tuple_type.hpp' namespace 'dynd::ndt':
    cppclass tuple_type:
        pass

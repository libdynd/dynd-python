from ..type cimport type
from ..array cimport array
from libcpp.vector cimport vector

cdef extern from 'dynd/types/callable_type.hpp' namespace 'dynd::ndt':
    cppclass callable_type:
        pass

from libc.stdint cimport intptr_t

from ..type cimport type

from ...config cimport translate_exception

cdef extern from 'dynd/types/base_fixed_dim_type.hpp' namespace 'dynd::ndt':
    cppclass base_fixed_dim_type:
        pass

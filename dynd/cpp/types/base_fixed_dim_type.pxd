from libc.stdint cimport intptr_t

from ..type cimport type

from ...config cimport translate_exception

cdef extern from 'dynd/types/fixed_dim_kind_type.hpp' namespace 'dynd::ndt':
    cppclass fixed_dim_kind_type:
        pass

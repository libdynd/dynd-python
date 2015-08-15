from libc.stdint cimport intptr_t

from ..type cimport type

from ...config cimport translate_exception

cdef extern from 'dynd/types/fixed_bytes_type.hpp' namespace 'dynd::ndt':
    type make_fixed_bytes(intptr_t, intptr_t) except +translate_exception

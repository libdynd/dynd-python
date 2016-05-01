from libc.stdint cimport intptr_t

cdef extern from "dynd/types/fixed_bytes_type.hpp" namespace "dynd::ndt" nogil:
    cppclass fixed_bytes_type:
        fixed_bytes_type(intptr_t element_size, intptr_t alignment)


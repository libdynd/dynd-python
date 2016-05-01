cdef extern from "dynd/types/bytes_type.hpp" namespace "dynd::ndt" nogil:
    cppclass bytes_type:
        bytes_type()
        bytes_type(size_t alignment)

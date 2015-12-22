cdef extern from "dynd/types/string_type.hpp" namespace "dynd::ndt" nogil:
    cppclass string_type:
        string_type()

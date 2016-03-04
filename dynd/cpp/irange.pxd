from ..config cimport translate_exception

cdef extern from "dynd/irange.hpp" namespace "dynd" nogil:
    cppclass irange:
        irange()
        # Use size_t for this interface since Cython doesn't know
        # how to directly convert integer literals in the source
        # to intptr_t. It can do that with size_t and on the overwhelming
        # majority of platforms that's exactly the same thing.
        irange(size_t)
        irange(size_t, size_t)
        irange(size_t, size_t, size_t)

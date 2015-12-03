from ..cpp.array cimport array as _array

cdef class array(object):
    cdef _array v

cpdef array asarray(obj, access=*)

cdef _array array_to_cpp(array) except *
cdef array array_from_cpp(_array)

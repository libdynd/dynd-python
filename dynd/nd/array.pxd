from ..cpp.array cimport array as _array

cdef class array(object):
    cdef _array v

cpdef array asarray(obj, access=*)

from ..cpp.array cimport array as _array

cdef class array(object):
    cdef _array v
    cdef inline _array as_cpp(array self):
        return self.v
    @staticmethod
    cdef array from_cpp(_array v)

cpdef array asarray(obj, access=*)

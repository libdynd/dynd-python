from ..cpp.type cimport type as _type

cdef class type(object):
    cdef _type v

cdef object as_numba_type(_type)
cdef _type from_numba_type(object)

# Should not have to repeat this declaration here, but doing it to get around a circular import
from ..cpp.array cimport array as _array
cdef class array(object):
    cdef _array v

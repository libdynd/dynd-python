from ..cpp.type cimport type as _type

cdef class type(object):
    cdef _type v

cdef object as_numba_type(_type)
cdef _type from_numba_type(object)

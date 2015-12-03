from ..cpp.type cimport type as _type

cdef api class type(object)[object dynd_ndt_type_pywrapper, type dynd_ndt_type_pywrapper_type]:
    cdef _type v

cdef api _type type_to_cpp(type) except *
cdef api type type_from_cpp(_type)

cdef object as_numba_type(_type)
cdef _type from_numba_type(object)

# Should not have to repeat this declaration here, but doing it to get around a circular import
from ..cpp.array cimport array as _array
cdef class array(object):
    cdef _array v

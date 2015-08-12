from libcpp.string cimport string

from ..config cimport translate_exception
from ..cpp.array cimport array as _array
from ..cpp.type cimport type as _type

cdef class type(object):
    cdef _type v

cdef as_numba_type(_type)
cdef _type from_numba_type(object)

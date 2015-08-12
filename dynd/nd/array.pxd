from libcpp.string cimport string

from ..config cimport translate_exception
from ..cpp.array cimport array as _array
from ..cpp.type cimport type as _type

cdef class array(object):
    cdef _array v

cpdef array asarray(obj, access=*)

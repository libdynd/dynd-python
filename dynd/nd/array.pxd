from libc.stdint cimport intptr_t
from libcpp.string cimport string

from ..config cimport translate_exception

from dynd.ndt.type cimport _type

cdef class array(object):
    cdef _array v

cpdef array asarray(obj, access=*)

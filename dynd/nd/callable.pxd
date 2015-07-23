from ..config cimport translate_exception
from .array cimport array
from dynd.ndt.type cimport _type

cdef extern from 'dynd/func/callable.hpp' namespace 'dynd':
    cdef cppclass _callable 'dynd::nd::callable':
        _type get_array_type()

cdef extern from "arrfunc_functions.hpp" namespace "pydynd":
    void init_w_callable_typeobject(object)
    object callable_call(object, object, object, object) except +translate_exception

cdef class callable(object):
    cdef _callable v
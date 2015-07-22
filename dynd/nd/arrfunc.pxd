from ..config cimport translate_exception
from .array cimport array
from dynd.ndt.type cimport _type

cdef extern from 'dynd/func/arrfunc.hpp' namespace 'dynd':
    cdef cppclass _arrfunc 'dynd::nd::arrfunc':
        _type get_array_type()

cdef extern from "arrfunc_functions.hpp" namespace "pydynd":
    void init_w_arrfunc_typeobject(object)
    object arrfunc_call(object, object, object, object) except +translate_exception

    object wrap_arrfunc(_arrfunc &) except +translate_exception

cdef class arrfunc(object):
    cdef _arrfunc v
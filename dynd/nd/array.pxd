from ..config cimport translate_exception

from dynd.ndt.type cimport _type

cdef extern from 'dynd/array.hpp' namespace 'dynd':
    cdef cppclass _array 'dynd::nd::array':

        _type get_type()

cdef extern from 'array_functions.hpp' namespace 'pydynd':
    void init_w_array_typeobject(object)

    void array_init_from_pyobject(_array&, object, object, bint, object) except +translate_exception
    void array_init_from_pyobject(_array&, object, object) except +translate_exception

    object array_as_py(_array&, bint) except +translate_exception

cdef extern from 'gfunc_callable_functions.hpp' namespace 'pydynd':
    object get_array_dynamic_property(_array&, object) except +translate_exception

cdef class array(object):
    cdef _array v
from libc.stdint cimport intptr_t
from libcpp.string cimport string

from ..config cimport translate_exception

from dynd.ndt.type cimport _type

cdef extern from 'dynd/array.hpp' namespace 'dynd':
    cdef cppclass _array 'dynd::nd::array':

        _type get_type()
        intptr_t get_dim_size() except +translate_exception

cdef extern from 'dynd/types/datashape_formatter.hpp' namespace 'dynd':
    string dynd_format_datashape 'dynd::format_datashape' (_array&) except +translate_exception

cdef extern from 'array_functions.hpp' namespace 'pydynd':
    void init_w_array_typeobject(object)

    void array_init_from_pyobject(_array&, object, object, bint, object) except +translate_exception
    void array_init_from_pyobject(_array&, object, object) except +translate_exception

    _array array_asarray(object, object) except +translate_exception
    object array_get_shape(_array&) except +translate_exception
    _array array_getitem(_array&, object) except +translate_exception
    void array_setitem(_array&, object, object) except +translate_exception
    _array array_view(object, object, object) except +translate_exception
    _array array_zeros(_type&, object) except +translate_exception
    _array array_zeros(object, _type&, object) except +translate_exception
    _array array_ones(_type&, object) except +translate_exception
    _array array_ones(object, _type&, object) except +translate_exception
    _array array_full(_type&, object, object) except +translate_exception
    _array array_full(object, _type&, object, object) except +translate_exception
    _array array_empty(_type&, object) except +translate_exception
    _array array_empty(object, _type&, object) except +translate_exception

    object array_as_py(_array&, bint) except +translate_exception

    const char *array_access_flags_string(_array&) except +translate_exception

cdef extern from 'gfunc_callable_functions.hpp' namespace 'pydynd':
    object get_array_dynamic_property(_array&, object) except +translate_exception

cdef class array(object):
    cdef _array v
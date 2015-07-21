from libc.stdint cimport intptr_t
from libcpp.string cimport string

from ..config cimport translate_exception

from dynd.ndt.type cimport _type

cdef extern from 'dynd/array.hpp' namespace 'dynd':
    cdef cppclass _array 'dynd::nd::array':

        _type get_type()
        _type get_dtype()
        _type get_dtype(size_t)
        intptr_t get_ndim()
        intptr_t get_dim_size() except +translate_exception

        _array view_scalars(_type&) except +translate_exception

    _array dynd_groupby "dynd::nd::groupby"(_array&, _array&, _type) except +translate_exception
    _array dynd_groupby "dynd::nd::groupby"(_array&, _array&) except +translate_exception

cdef extern from 'dynd/types/datashape_formatter.hpp' namespace 'dynd':
    string dynd_format_datashape 'dynd::format_datashape' (_array&) except +translate_exception

cdef extern from 'array_functions.hpp' namespace 'pydynd':
    void init_w_array_typeobject(object)

    void array_init_from_pyobject(_array&, object, object, bint, object) except +translate_exception
    void array_init_from_pyobject(_array&, object, object) except +translate_exception

    object array_int(_array&) except +translate_exception
    object array_float(_array&) except +translate_exception
    object array_complex(_array&) except +translate_exception

    _array array_asarray(object, object) except +translate_exception
    object array_get_shape(_array&) except +translate_exception
    object array_get_strides(_array&) except +translate_exception
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
    object array_index(_array&) except +translate_exception
    bint array_contains(_array&, object) except +translate_exception
    object array_nonzero(_array&) except +translate_exception

    _array array_eval(_array&, object) except +translate_exception
    _array array_cast(_array&, _type&) except +translate_exception
    _array array_ucast(_array&, _type&, size_t) except +translate_exception
    _array array_range(object, object, object, object) except +translate_exception
    _array array_linspace(object, object, object, object) except +translate_exception

    bint array_is_c_contiguous(_array&) except +translate_exception
    bint array_is_f_contiguous(_array&) except +translate_exception

    object array_as_py(_array&, bint) except +translate_exception
    object array_as_numpy(object, bint) except +translate_exception

    const char *array_access_flags_string(_array&) except +translate_exception

    string array_repr(_array&) except +translate_exception
    object array_str(_array&) except +translate_exception
    object array_unicode(_array&) except +translate_exception

    _array array_add(_array&, _array&) except +translate_exception
    _array array_subtract(_array&, _array&) except +translate_exception
    _array array_multiply(_array&, _array&) except +translate_exception
    _array array_divide(_array&, _array&) except +translate_exception

    _array dynd_parse_json_type(_type&, _array&, object) except +translate_exception
    void dynd_parse_json_array(_array&, _array&, object) except +translate_exception

    _array nd_fields(_array&, object) except +translate_exception

    int array_getbuffer_pep3118(object ndo, Py_buffer *buffer, int flags) except -1
    int array_releasebuffer_pep3118(object ndo, Py_buffer *buffer) except -1

cdef extern from 'gfunc_callable_functions.hpp' namespace 'pydynd':
    object get_array_dynamic_property(_array&, object) except +translate_exception
    void add_array_names_to_dir_dict(_array&, object) except +translate_exception
    void set_array_dynamic_property(_array&, object, object) except +translate_exception

    cdef cppclass array_callable_wrapper:
        pass
    object array_callable_call(array_callable_wrapper&, object, object) except +translate_exception
    void init_w_array_callable_typeobject(object)

cdef class array(object):
    cdef _array v
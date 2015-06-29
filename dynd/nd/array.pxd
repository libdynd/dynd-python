#
# Copyright (C) 2011-15 DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

from translate_except cimport translate_exception

cdef extern from "<complex>" namespace "std":
    cdef cppclass complex[T]:
        T real()
        T imag()

from type cimport ndt_type

from libc.stdint cimport intptr_t, uintptr_t
from libcpp.string cimport string

cdef extern from "<iostream>" namespace "std":
    cdef cppclass ostream:
        pass

    extern ostream cout

cdef extern from 'dynd/array.hpp' namespace 'dynd':
    cdef cppclass _array 'dynd::nd::array':
        _array() except +translate_exception
        _array(signed char value)
        _array(short value)
        _array(int value)
        _array(long value)
        _array(long long value)
        _array(unsigned char value)
        _array(unsigned short value)
        _array(unsigned int value)
        _array(unsigned long value)
        _array(unsigned long long value)
        _array(float value)
        _array(double value)
        _array(complex[float] value)
        _array(complex[double] value)
        _array(ndt_type&)
        _array(ndt_type, int, intptr_t *, int *)

        # Cython bug: operator overloading doesn't obey "except +"
        # TODO: Report this bug
        # _array operator+(_array&) except +translate_exception
        #_array operator-(_array&) except +translate_exception
        #_array operator*(_array&) except +translate_exception
        #_array operator/(_array&) except +translate_exception

        ndt_type get_type()
        ndt_type get_dtype()
        ndt_type get_dtype(size_t)
        intptr_t get_ndim()
        bint is_scalar()
        intptr_t get_dim_size() except +translate_exception

        char* get_readwrite_originptr()
        char* get_readonly_originptr()

        _array vals() except +translate_exception

        _array eval() except +translate_exception
        _array eval_immutable() except +translate_exception

        _array storage() except +translate_exception

        _array view_scalars(ndt_type&) except +translate_exception
        _array ucast(ndt_type&, size_t) except +translate_exception

        void flag_as_immutable() except +translate_exception

        void debug_print(ostream&)

    _array dynd_groupby "dynd::nd::groupby"(_array&, _array&, ndt_type) except +translate_exception
    _array dynd_groupby "dynd::nd::groupby"(_array&, _array&) except +translate_exception

cdef extern from "array_functions.hpp" namespace "pydynd":
    void init_w_array_typeobject(object)

    string array_repr(_array&) except +translate_exception
    object array_str(_array&) except +translate_exception
    object array_unicode(_array&) except +translate_exception
    object array_index(_array&) except +translate_exception
    object array_nonzero(_array&) except +translate_exception
    object array_int(_array&) except +translate_exception
    object array_float(_array&) except +translate_exception
    object array_complex(_array&) except +translate_exception
    string array_debug_print(_array&) except +translate_exception
    bint array_contains(_array&, object) except +translate_exception

    void array_init_from_pyobject(_array&, object, object, bint, object) except +translate_exception
    void array_init_from_pyobject(_array&, object, object) except +translate_exception
    _array array_view(object, object, object) except +translate_exception
    _array array_asarray(object, object) except +translate_exception
    _array array_eval(_array&, object) except +translate_exception
    _array array_eval_copy(_array&, object, object) except +translate_exception
    _array array_zeros(ndt_type&, object) except +translate_exception
    _array array_zeros(object, ndt_type&, object) except +translate_exception
    _array array_ones(ndt_type&, object) except +translate_exception
    _array array_ones(object, ndt_type&, object) except +translate_exception
    _array array_full(ndt_type&, object, object) except +translate_exception
    _array array_full(object, ndt_type&, object, object) except +translate_exception
    _array array_empty(ndt_type&, object) except +translate_exception
    _array array_empty(object, ndt_type&, object) except +translate_exception
    _array array_empty_like(_array&) except +translate_exception
    _array array_empty_like(_array&, ndt_type&) except +translate_exception
    _array array_memmap(object, object, object, object) except +translate_exception

    _array array_add(_array&, _array&) except +translate_exception
    _array array_subtract(_array&, _array&) except +translate_exception
    _array array_multiply(_array&, _array&) except +translate_exception
    _array array_divide(_array&, _array&) except +translate_exception

    _array array_getitem(_array&, object) except +translate_exception
    void array_setitem(_array&, object, object) except +translate_exception
    object array_get_shape(_array&) except +translate_exception
    object array_get_strides(_array&) except +translate_exception
    bint array_is_scalar(_array&) except +translate_exception

    bint array_is_c_contiguous(_array&) except +translate_exception
    bint array_is_f_contiguous(_array&) except +translate_exception

    _array array_range(object, object, object, object) except +translate_exception
    _array array_linspace(object, object, object, object) except +translate_exception
    _array nd_fields(_array&, object) except +translate_exception

    _array array_cast(_array&, ndt_type&) except +translate_exception
    _array array_ucast(_array&, ndt_type&, size_t) except +translate_exception
    object array_adapt(object, object, object) except +translate_exception
    object array_as_py(_array&, bint) except +translate_exception
    object array_as_numpy(object, bint) except +translate_exception
    _array array_from_py(object) except +translate_exception

    int array_getbuffer_pep3118(object ndo, Py_buffer *buffer, int flags) except -1
    int array_releasebuffer_pep3118(object ndo, Py_buffer *buffer) except -1

    const char *array_access_flags_string(_array&) except +translate_exception

    _array dynd_parse_json_type(ndt_type&, _array&, object) except +translate_exception
    void dynd_parse_json_array(_array&, _array&, object) except +translate_exception

    object wrap_array(const _array &af)
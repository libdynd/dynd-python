#
# Copyright (C) 2011-13 Mark Wiebe, DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

cdef extern from "dynd/array.hpp" namespace "dynd":
    cdef cppclass ndarray "dynd::nd::array":
        ndarray() except +translate_exception
        ndarray(signed char value)
        ndarray(short value)
        ndarray(int value)
        ndarray(long value)
        ndarray(long long value)
        ndarray(unsigned char value)
        ndarray(unsigned short value)
        ndarray(unsigned int value)
        ndarray(unsigned long value)
        ndarray(unsigned long long value)
        ndarray(float value)
        ndarray(double value)
        ndarray(complex[float] value)
        ndarray(complex[double] value)
        ndarray(ndt_type&)
        ndarray(ndt_type, int, intptr_t *, int *)

        # Cython bug: operator overloading doesn't obey "except +"
        # TODO: Report this bug
        # ndarray operator+(ndarray&) except +translate_exception
        #ndarray operator-(ndarray&) except +translate_exception
        #ndarray operator*(ndarray&) except +translate_exception
        #ndarray operator/(ndarray&) except +translate_exception

        ndt_type get_type()
        ndt_type get_dtype()
        ndt_type get_dtype(size_t)
        size_t get_ndim()
        bint is_scalar()
        intptr_t get_dim_size()

        char* get_readwrite_originptr()
        char* get_readonly_originptr()

        ndarray vals() except +translate_exception

        ndarray eval() except +translate_exception
        ndarray eval_immutable() except +translate_exception

        ndarray storage() except +translate_exception

        ndarray view_scalars(ndt_type&) except +translate_exception
        ndarray ucast(ndt_type&, size_t, assign_error_mode) except +translate_exception

        void flag_as_immutable() except +translate_exception

        void debug_print(ostream&)

    ndarray dynd_groupby "dynd::nd::groupby" (ndarray&, ndarray&, ndt_type) except +translate_exception
    ndarray dynd_groupby "dynd::nd::groupby" (ndarray&, ndarray&) except +translate_exception

cdef extern from "array_functions.hpp" namespace "pydynd":
    void init_w_array_typeobject(object)

    object array_str(ndarray&) except +translate_exception
    object array_unicode(ndarray&) except +translate_exception
    object array_index(ndarray&) except +translate_exception
    object array_nonzero(ndarray&) except +translate_exception
    string array_repr(ndarray&) except +translate_exception
    string array_debug_print(ndarray&) except +translate_exception
    bint array_contains(ndarray&, object) except +translate_exception

    void array_init_from_pyobject(ndarray&, object, object, bint, object) except +translate_exception
    void array_init_from_pyobject(ndarray&, object, object) except +translate_exception
    ndarray array_view(object, object) except +translate_exception
    ndarray array_asarray(object, object) except +translate_exception
    ndarray array_eval(ndarray&) except +translate_exception
    ndarray array_eval_copy(ndarray&, object) except +translate_exception
    ndarray array_empty(ndt_type&) except +translate_exception
    ndarray array_empty(object, ndt_type&) except +translate_exception
    ndarray array_empty_like(ndarray&) except +translate_exception
    ndarray array_empty_like(ndarray&, ndt_type&) except +translate_exception

    ndarray array_add(ndarray&, ndarray&) except +translate_exception
    ndarray array_subtract(ndarray&, ndarray&) except +translate_exception
    ndarray array_multiply(ndarray&, ndarray&) except +translate_exception
    ndarray array_divide(ndarray&, ndarray&) except +translate_exception

    ndarray array_getitem(ndarray&, object) except +translate_exception
    void array_setitem(ndarray&, object, object) except +translate_exception
    object array_get_shape(ndarray&) except +translate_exception
    object array_get_strides(ndarray&) except +translate_exception

    ndarray array_range(object, object, object, object) except +translate_exception
    ndarray array_linspace(object, object, object, object) except +translate_exception
    ndarray nd_fields(ndarray&, object) except +translate_exception

    ndarray array_cast(ndarray&, ndt_type&, object) except +translate_exception
    ndarray array_ucast(ndarray&, ndt_type&, size_t, object) except +translate_exception
    object array_as_py(ndarray&) except +translate_exception
    object array_as_numpy(object, bint) except +translate_exception
    ndarray array_from_py(object) except +translate_exception

    int array_getbuffer_pep3118(object ndo, Py_buffer *buffer, int flags) except -1
    int array_releasebuffer_pep3118(object ndo, Py_buffer *buffer) except -1

    const char *array_access_flags_string(ndarray&) except +translate_exception
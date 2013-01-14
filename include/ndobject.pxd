#
# Copyright (C) 2011-12, Dynamic NDArray Developers
# BSD 2-Clause License, see LICENSE.txt
#

cdef extern from "dynd/ndobject.hpp" namespace "dynd":
    cdef cppclass ndobject:
        ndobject() except +translate_exception
        ndobject(signed char value)
        ndobject(short value)
        ndobject(int value)
        ndobject(long value)
        ndobject(long long value)
        ndobject(unsigned char value)
        ndobject(unsigned short value)
        ndobject(unsigned int value)
        ndobject(unsigned long value)
        ndobject(unsigned long long value)
        ndobject(float value)
        ndobject(double value)
        ndobject(complex[float] value)
        ndobject(complex[double] value)
        ndobject(dtype&)
        ndobject(dtype, int, intptr_t *, int *)

        # Cython bug: operator overloading doesn't obey "except +"
        # TODO: Report this bug
        # ndobject operator+(ndobject&) except +translate_exception
        #ndobject operator-(ndobject&) except +translate_exception
        #ndobject operator*(ndobject&) except +translate_exception
        #ndobject operator/(ndobject&) except +translate_exception

        dtype get_dtype()
        bint is_scalar()
        intptr_t get_dim_size()

        char* get_readwrite_originptr()
        char* get_readonly_originptr()

        void val_assign(ndobject&, assign_error_mode) except +translate_exception
        void val_assign(dtype&, char*, assign_error_mode) except +translate_exception

        ndobject vals() except +translate_exception

        ndobject eval_immutable() except +translate_exception

        ndobject storage() except +translate_exception

        ndobject cast_scalars(dtype&, assign_error_mode) except +translate_exception

        ndobject view_scalars(dtype&) except +translate_exception

        void debug_print(ostream&)

    ndobject dynd_groupby "dynd::groupby" (ndobject, ndobject, dtype) except +translate_exception

cdef extern from "ndobject_functions.hpp" namespace "pydynd":
    void init_w_ndobject_typeobject(object)

    string ndobject_str(ndobject&) except +translate_exception
    string ndobject_repr(ndobject&) except +translate_exception
    string ndobject_debug_print(ndobject&) except +translate_exception

    void ndobject_init_from_pyobject(ndobject&, object obj) except +translate_exception
    ndobject ndobject_vals(ndobject&) except +translate_exception
    ndobject ndobject_eval_copy(ndobject&, object) except +translate_exception
    ndobject ndobject_empty_like(ndobject&) except +translate_exception
    ndobject ndobject_empty_like(ndobject&, dtype&) except +translate_exception

    ndobject ndobject_add(ndobject&, ndobject&) except +translate_exception
    ndobject ndobject_subtract(ndobject&, ndobject&) except +translate_exception
    ndobject ndobject_multiply(ndobject&, ndobject&) except +translate_exception
    ndobject ndobject_divide(ndobject&, ndobject&) except +translate_exception

    ndobject ndobject_getitem(ndobject&, object) except +translate_exception
    object ndobject_get_shape(ndobject&) except +translate_exception
    object ndobject_get_strides(ndobject&) except +translate_exception

    ndobject ndobject_arange(object, object, object) except +translate_exception
    ndobject ndobject_linspace(object, object, object) except +translate_exception

    ndobject ndobject_cast_scalars(ndobject&, dtype& , object) except +translate_exception
    ndobject ndobject_cast_udtype(ndobject&, dtype& , object) except +translate_exception
    object ndobject_as_py(ndobject&) except +translate_exception
    ndobject ndobject_from_py(object) except +translate_exception

    int ndobject_getbuffer_pep3118(object ndo, Py_buffer *buffer, int flags) except -1
    int ndobject_releasebuffer_pep3118(object ndo, Py_buffer *buffer) except -1

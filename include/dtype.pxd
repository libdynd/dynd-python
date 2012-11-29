#
# Copyright (C) 2011-12, Dynamic NDArray Developers
# BSD 2-Clause License, see LICENSE.txt
#

cdef extern from "dynd/dtype.hpp" namespace "dynd":
    cdef enum dtype_kind_t:
        bool_kind
        int_kind
        uint_kind
        real_kind
        complex_kind
        string_kind
        composite_kind
        expression_kind
        pattern_kind
        custom_kind

    cdef enum type_id_t:
        bool_type_id
        int8_type_id
        int16_type_id
        int32_type_id
        int64_type_id
        uint8_type_id
        uint16_type_id
        uint32_type_id
        uint64_type_id
        float32_type_id
        float64_type_id
        complex_float32_type_id
        complex_float64_type_id
        utf8_type_id
        struct_type_id
        tuple_type_id
        array_type_id
        ndobject_type_id
        convert_type_id
        pattern_type_id

    cdef enum string_encoding_t:
        string_encoding_ascii
        string_encoding_ucs_2
        string_encoding_utf_8
        string_encoding_utf_16
        string_encoding_utf_32
        string_encoding_invalid

    cdef cppclass extended_dtype:
        type_id_t type_id()
        dtype_kind_t get_kind()
        uintptr_t get_alignment()
        uintptr_t get_element_size()
        dtype& value_dtype(dtype&)
        dtype& operand_dtype(dtype&)

    cdef cppclass dtype:
        dtype()
        dtype(type_id_t) except +translate_exception
        dtype(type_id_t, uintptr_t) except +translate_exception
        dtype(string&) except +translate_exception
        bint operator==(dtype&)
        bint operator!=(dtype&)
        
        dtype& value_dtype()
        dtype& operand_dtype()
        dtype& storage_dtype()
        dtype get_canonical_dtype()
        type_id_t type_id()
        dtype_kind_t get_kind()
        int get_alignment()
        uintptr_t get_element_size()
        extended_dtype* extended()
        string_encoding_t string_encoding() except +translate_exception
        int get_uniform_ndim()

cdef extern from "dynd/dtype_assign.hpp" namespace "dynd":
    cdef enum assign_error_mode:
        assign_error_none
        assign_error_overflow
        assign_error_fractional
        assign_error_inexact
        assign_error_default

cdef extern from "dynd/dtypes/fixedbytes_dtype.hpp" namespace "dynd":
    dtype dnd_make_fixedbytes_dtype "dynd::make_fixedbytes_dtype" (intptr_t, intptr_t) except +translate_exception

cdef extern from "dynd/dtypes/byteswap_dtype.hpp" namespace "dynd":
    dtype dnd_make_byteswap_dtype "dynd::make_byteswap_dtype" (dtype&) except +translate_exception
    dtype dnd_make_byteswap_dtype "dynd::make_byteswap_dtype" (dtype&, dtype&) except +translate_exception

cdef extern from "dynd/dtypes/categorical_dtype.hpp" namespace "dynd":
    dtype dnd_make_categorical_dtype "dynd::make_categorical_dtype" (ndobject&) except +translate_exception
    dtype dnd_factor_categorical_dtype "dynd::factor_categorical_dtype" (ndobject&) except +translate_exception

cdef extern from "dynd/dtypes/dtype_alignment.hpp" namespace "dynd":
    dtype dnd_make_unaligned_dtype "dynd::make_unaligned_dtype" (dtype&) except +translate_exception

cdef extern from "dtype_functions.hpp" namespace "pydynd":
    void init_w_dtype_typeobject(object)

    string dtype_str(dtype&)
    string dtype_repr(dtype&)
    dtype deduce_dtype_from_object(object) except +translate_exception
    dtype make_dtype_from_object(object) except +translate_exception

    object dtype_get_kind(dtype&) except +translate_exception
    dtype dtype_getitem(dtype&, object) except +translate_exception

    dtype dnd_make_convert_dtype(dtype&, dtype&, object) except +translate_exception
    dtype dnd_make_fixedstring_dtype(object, int) except +translate_exception
    dtype dnd_make_string_dtype(object) except +translate_exception
    dtype dnd_make_pointer_dtype(dtype&) except +translate_exception

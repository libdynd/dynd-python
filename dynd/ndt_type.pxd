#
# Copyright (C) 2011-15 DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

from libc.stdint cimport uintptr_t, intptr_t
from libcpp.string cimport string

from translate_except cimport translate_exception

cdef extern from "dynd/type.hpp" namespace "dynd::ndt":
    cdef enum type_kind_t:
        bool_kind
        sint_kind
        real_kind
        complex_kind
        string_kind
        composite_kind
        expr_kind
        symbolic_kind
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
        ndarray_type_id
        convert_type_id
        pattern_type_id

    cdef enum string_encoding_t:
        string_encoding_ascii
        string_encoding_ucs_2
        string_encoding_utf_8
        string_encoding_utf_16
        string_encoding_utf_32
        string_encoding_invalid

    cdef cppclass base_type:
        type_id_t type_id()
        type_kind_t get_kind()
        uintptr_t get_data_size()
        uintptr_t get_data_alignment()
        ndt_type& value_type(ndt_type&)
        ndt_type& operand_type(ndt_type&)

    cdef cppclass ndt_type "dynd::ndt::type":
        ndt_type()
        ndt_type(type_id_t) except +translate_exception
        ndt_type(type_id_t, uintptr_t) except +translate_exception
        ndt_type(string&) except +translate_exception
        bint operator==(ndt_type&)
        bint operator!=(ndt_type&)

        ndt_type& value_type()
        ndt_type& operand_type()
        ndt_type& storage_type()
        ndt_type get_canonical_type()
        type_id_t type_id()
        type_kind_t get_kind()
        size_t get_data_size()
        size_t get_default_data_size() except +translate_exception
        size_t get_data_alignment()
        size_t get_arrmeta_size()
        base_type* extended()
        string_encoding_t string_encoding() except +translate_exception
        intptr_t get_ndim()
        ndt_type get_dtype() except +translate_exception
        ndt_type get_dtype(size_t) except +translate_exception

        ndt_type with_replaced_dtype(ndt_type&, size_t) except +translate_exception

        bint match(ndt_type&) except +translate_exception

cdef extern from "dynd/typed_data_assign.hpp" namespace "dynd":
    cdef enum assign_error_mode:
        assign_error_nocheck
        assign_error_overflow
        assign_error_fractional
        assign_error_inexact
        assign_error_default

from array cimport ndarray

cdef extern from "dynd/types/fixed_bytes_type.hpp" namespace "dynd":
    ndt_type dynd_make_fixed_bytes_type "dynd::ndt::make_fixed_bytes" (intptr_t, intptr_t) except +translate_exception

cdef extern from "dynd/types/byteswap_type.hpp" namespace "dynd":
    ndt_type dynd_make_byteswap_type "dynd::ndt::make_byteswap" (ndt_type&) except +translate_exception
    ndt_type dynd_make_byteswap_type "dynd::ndt::make_byteswap" (ndt_type&, ndt_type&) except +translate_exception

cdef extern from "dynd/types/categorical_type.hpp" namespace "dynd":
    ndt_type dynd_make_categorical_type "dynd::ndt::make_categorical" (ndarray&) except +translate_exception
    ndt_type dynd_factor_categorical_type "dynd::ndt::factor_categorical" (ndarray&) except +translate_exception

cdef extern from "dynd/types/type_alignment.hpp" namespace "dynd":
    ndt_type dynd_make_unaligned_type "dynd::ndt::make_unaligned" (ndt_type&) except +translate_exception

cdef extern from "dynd/types/fixed_dim_kind_type.hpp" namespace "dynd":
    ndt_type dynd_make_fixed_dim_kind_type "dynd::ndt::make_fixed_dim_kind" (ndt_type&) except +translate_exception
    ndt_type dynd_make_fixed_dim_kind_type "dynd::ndt::make_fixed_dim_kind" (ndt_type&, intptr_t) except +translate_exception

cdef extern from "dynd/types/pow_dimsym_type.hpp" namespace "dynd":
    ndt_type dynd_make_pow_dimsym_type "dynd::ndt::make_pow_dimsym" (ndt_type&, string&, ndt_type&) except +translate_exception

cdef extern from "dynd/types/var_dim_type.hpp" namespace "dynd":
    ndt_type dynd_make_var_dim_type "dynd::ndt::make_var_dim" (ndt_type&) except +translate_exception

cdef extern from "dynd/types/property_type.hpp" namespace "dynd":
    ndt_type dynd_make_property_type "dynd::ndt::make_property" (ndt_type&, string) except +translate_exception
    ndt_type dynd_make_reversed_property_type "dynd::ndt::make_reversed_property" (ndt_type&, ndt_type&, string&) except +translate_exception

cdef extern from "dynd/types/bytes_type.hpp" namespace "dynd":
    ndt_type dynd_make_bytes_type "dynd::ndt::make_bytes" (size_t) except +translate_exception

cdef extern from "type_functions.hpp" namespace "pydynd":
    void init_w_type_typeobject(object)

    string ndt_type_str(ndt_type&)
    string ndt_type_repr(ndt_type&)
    ndt_type make_ndt_type_from_pyobject(object) except +translate_exception

    object ndt_type_get_shape(ndt_type&) except +translate_exception
    object ndt_type_get_kind(ndt_type&) except +translate_exception
    object ndt_type_get_type_id(ndt_type&) except +translate_exception
    ndt_type ndt_type_getitem(ndt_type&, object) except +translate_exception
    object ndt_type_array_property_names(ndt_type&) except +translate_exception

    ndt_type dynd_make_convert_type(ndt_type&, ndt_type&) except +translate_exception
    ndt_type dynd_make_view_type(ndt_type&, ndt_type&) except +translate_exception
    ndt_type dynd_make_fixed_string_type(int, object) except +translate_exception
    ndt_type dynd_make_string_type(object) except +translate_exception
    ndt_type dynd_make_pointer_type(ndt_type&) except +translate_exception
    ndt_type dynd_make_struct_type(object, object) except +translate_exception
    ndt_type dynd_make_cstruct_type(object, object) except +translate_exception
    ndt_type dynd_make_fixed_dim_type(object, ndt_type&) except +translate_exception
    ndt_type dynd_make_cfixed_dim_type(object, ndt_type&, object) except +translate_exception

cdef extern from "numpy_interop.hpp" namespace "pydynd":
    object numpy_dtype_obj_from_ndt_type(ndt_type&) except +translate_exception

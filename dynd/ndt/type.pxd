#
# Copyright (C) 2011-15 DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

from libc.stdint cimport uintptr_t, intptr_t
from libcpp.string cimport string

from translate_except cimport translate_exception

cdef extern from 'dynd/type.hpp' namespace 'dynd::ndt':
    cdef enum type_kind_t:
        bool_kind
        uint_kind
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
        _type& value_type(_type&)
        _type& operand_type(_type&)

    cdef cppclass _type 'dynd::ndt::type':
        _type()
        _type(type_id_t) except +translate_exception
        _type(type_id_t, uintptr_t) except +translate_exception
        _type(string&) except +translate_exception
        bint operator==(_type&)
        bint operator!=(_type&)

        _type& value_type()
        _type& operand_type()
        _type& storage_type()
        _type get_canonical_type()
        type_id_t type_id()
        type_kind_t get_kind()
        size_t get_data_size()
        size_t get_default_data_size() except +translate_exception
        size_t get_data_alignment()
        size_t get_arrmeta_size()
        base_type* extended()
        string_encoding_t string_encoding() except +translate_exception
        intptr_t get_ndim()
        _type get_dtype() except +translate_exception
        _type get_dtype(size_t) except +translate_exception

        _type with_replaced_dtype(_type&, size_t) except +translate_exception

        bint match(_type&) except +translate_exception

cdef extern from "dynd/typed_data_assign.hpp" namespace "dynd":
    cdef enum assign_error_mode:
        assign_error_nocheck
        assign_error_overflow
        assign_error_fractional
        assign_error_inexact
        assign_error_default

from nd.array cimport _array

cdef extern from "dynd/types/fixed_bytes_type.hpp" namespace "dynd":
    _type dynd_make_fixed_bytes_type "dynd::ndt::make_fixed_bytes" (intptr_t, intptr_t) except +translate_exception

cdef extern from "dynd/types/byteswap_type.hpp" namespace "dynd":
    _type dynd_make_byteswap_type "dynd::ndt::byteswap_type::make" (_type&) except +translate_exception
    _type dynd_make_byteswap_type "dynd::ndt::byteswap_type::make" (_type&, _type&) except +translate_exception

cdef extern from "dynd/types/categorical_type.hpp" namespace "dynd":
    _type dynd_make_categorical_type "dynd::ndt::categorical_type::make" (_array&) except +translate_exception
    _type dynd_factor_categorical_type "dynd::ndt::factor_categorical" (_array&) except +translate_exception

cdef extern from "dynd/types/type_alignment.hpp" namespace "dynd":
    _type dynd_make_unaligned_type "dynd::ndt::make_unaligned" (_type&) except +translate_exception

cdef extern from "dynd/types/fixed_dim_kind_type.hpp" namespace "dynd":
    _type dynd_make_fixed_dim_kind_type "dynd::ndt::fixed_dim_kind_type::make" (_type&) except +translate_exception
    _type dynd_make_fixed_dim_kind_type "dynd::ndt::fixed_dim_kind_type::make" (_type&, intptr_t) except +translate_exception

cdef extern from "dynd/types/pow_dimsym_type.hpp" namespace "dynd":
    _type dynd_make_pow_dimsym_type "dynd::ndt::make_pow_dimsym" (_type&, string&, _type&) except +translate_exception

cdef extern from "dynd/types/var_dim_type.hpp" namespace "dynd":
    _type dynd_make_var_dim_type "dynd::ndt::var_dim_type::make" (_type&) except +translate_exception

cdef extern from "dynd/types/property_type.hpp" namespace "dynd":
    _type dynd_make_property_type "dynd::ndt::property_type::make" (_type&, string) except +translate_exception
    _type dynd_make_reversed_property_type "dynd::ndt::property_type::make_reversed" (_type&, _type&, string&) except +translate_exception

cdef extern from "dynd/types/bytes_type.hpp" namespace "dynd":
    _type dynd_make_bytes_type "dynd::ndt::bytes_type::make" (size_t) except +translate_exception

cdef extern from "type_functions.hpp" namespace "pydynd":
    void init_w_type_typeobject(object)

    string _type_str(_type&)
    string _type_repr(_type&)
    _type make__type_from_pyobject(object) except +translate_exception

    object _type_get_shape(_type&) except +translate_exception
    object _type_get_kind(_type&) except +translate_exception
    object _type_get_type_id(_type&) except +translate_exception
    _type _type_getitem(_type&, object) except +translate_exception
    object _type_array_property_names(_type&) except +translate_exception

    _type dynd_make_convert_type(_type&, _type&) except +translate_exception
    _type dynd_make_view_type(_type&, _type&) except +translate_exception
    _type dynd_make_fixed_string_type(int, object) except +translate_exception
    _type dynd_make_string_type(object) except +translate_exception
    _type dynd_make_pointer_type(_type&) except +translate_exception
    _type dynd_make_struct_type(object, object) except +translate_exception
    _type dynd_make_cstruct_type(object, object) except +translate_exception
    _type dynd_make_fixed_dim_type(object, _type&) except +translate_exception
    _type dynd_make_cfixed_dim_type(object, _type&, object) except +translate_exception

cdef extern from "numpy_interop.hpp" namespace "pydynd":
    object numpy_dtype_obj_from__type(_type&) except +translate_exception

#cdef extern from "<dynd/types/datashape_formatter.hpp>" namespace "dynd":
#    string dynd_format_datashape "dynd::format_datashape" (_type&) except +translate_exception


cdef extern from "gfunc_callable_functions.hpp" namespace "pydynd":
    void add__type_names_to_dir_dict(_type&, object) except +translate_exception
    object get__type_dynamic_property(_type&, object) except +translate_exception

    void add_array_names_to_dir_dict(_array&, object) except +translate_exception
    object get_array_dynamic_property(_array&, object) except +translate_exception
    void set_array_dynamic_property(_array&, object, object) except +translate_exception

    # Function properties
    cdef cppclass _type_callable_wrapper:
        pass
    object _type_callable_call(_type_callable_wrapper&, object, object) except +translate_exception
    object call__type_constructor_function(_type&, object, object) except +translate_exception

    void init_w__type_callable_typeobject(object)
    cdef struct _type_callable_placement_wrapper:
        pass

    # Function property of nd::array
    cdef cppclass array_callable_wrapper:
        pass
    object array_callable_call(array_callable_wrapper&, object, object) except +translate_exception

    void init_w_array_callable_typeobject(object)
    cdef struct array_callable_placement_wrapper:
        pass
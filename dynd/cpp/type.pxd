from libc.stdint cimport intptr_t
from libcpp.string cimport string

from ..config cimport translate_exception
from .array cimport array


cdef extern from 'dynd/types/type_id.hpp' namespace 'dynd':
    ctypedef enum type_id_t:
        uninitialized_type_id
        bool_type_id
        int8_type_id
        int16_type_id
        int32_type_id
        int64_type_id
        int128_type_id
        uint8_type_id
        uint16_type_id
        uint32_type_id
        uint64_type_id
        uint128_type_id
        float16_type_id
        float32_type_id
        float64_type_id
        float128_type_id
        complex_float32_type_id
        complex_float64_type_id
        void_type_id

        pointer_type_id
        void_pointer_type_id

        bytes_type_id
        fixed_bytes_type_id

        char_type_id
        string_type_id
        fixed_string_type_id

        categorical_type_id

        option_type_id

        date_type_id
        time_type_id
        datetime_type_id
        busdate_type_id

        json_type_id

        tuple_type_id
        struct_type_id

        fixed_dim_type_id
        offset_dim_type_id
        var_dim_type_id

        callable_type_id

        type_type_id

cdef extern from 'dynd/type.hpp' namespace 'dynd::ndt':
    cdef cppclass type 'dynd::ndt::type':
        type()
        type(type_id_t) except +translate_exception
        type(string&) except +translate_exception

        size_t get_data_size()
        size_t get_default_data_size() except +translate_exception
        size_t get_data_alignment()
        size_t get_arrmeta_size()
        type get_canonical_type()

        bint operator==(type&)
        bint operator!=(type&)

        bint match(type&) except +translate_exception

        type_id_t get_type_id() const

cdef extern from "dynd/types/type_alignment.hpp" namespace "dynd":
    type dynd_make_unaligned_type "dynd::ndt::make_unaligned" (type&) except +translate_exception

cdef extern from "dynd/types/byteswap_type.hpp" namespace "dynd":
    type dynd_make_byteswap_type "dynd::ndt::byteswap_type::make" (type&) except +translate_exception
    type dynd_make_byteswap_type "dynd::ndt::byteswap_type::make" (type&, type&) except +translate_exception

cdef extern from 'dynd/types/fixed_bytes_type.hpp' namespace 'dynd':
    type dynd_make_fixed_bytes_type 'dynd::ndt::make_fixed_bytes'(intptr_t, intptr_t) except +translate_exception

cdef extern from "dynd/types/fixed_dim_kind_type.hpp" namespace "dynd":
    type dynd_make_fixed_dim_kind_type "dynd::ndt::fixed_dim_kind_type::make" (type&) except +translate_exception
    type dynd_make_fixed_dim_kind_type "dynd::ndt::fixed_dim_kind_type::make" (type&, intptr_t) except +translate_exception

cdef extern from "dynd/types/var_dim_type.hpp" namespace "dynd":
    type dynd_make_var_dim_type "dynd::ndt::var_dim_type::make" (type&) except +translate_exception

cdef extern from 'dynd/types/datashape_formatter.hpp' namespace 'dynd':
    string dynd_format_datashape 'dynd::format_datashape' (type&) except +translate_exception

cdef extern from "dynd/types/categorical_type.hpp" namespace "dynd":
    type dynd_make_categorical_type "dynd::ndt::categorical_type::make" (array&) except +translate_exception
    type dynd_factor_categorical_type "dynd::ndt::factor_categorical" (array&) except +translate_exception

cdef extern from 'dynd/types/callable_type.hpp':
    type make_callable 'dynd::ndt::callable_type::make'(type &, array &)
    type make_callable 'dynd::ndt::callable_type::make'(type &, type &, type &)

cdef extern from 'dynd/types/tuple_type.hpp':
    type make_tuple 'dynd::ndt::tuple_type::make'() \
        except +translate_exception
    type make_tuple 'dynd::ndt::tuple_type::make'(const array &) \
        except +translate_exception

cdef extern from 'dynd/types/struct_type.hpp':
    type make_struct 'dynd::ndt::struct_type::make'() \
        except +translate_exception
    type make_struct 'dynd::ndt::struct_type::make'(const array &, const array &) \
        except +translate_exception

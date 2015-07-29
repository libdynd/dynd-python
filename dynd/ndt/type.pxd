from libc.stdint cimport intptr_t
from libcpp.string cimport string

from ..config cimport translate_exception
from dynd.nd.array cimport _array

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
    cdef cppclass _type 'dynd::ndt::type':
        _type()
        _type(type_id_t) except +translate_exception
        _type(string&) except +translate_exception

        size_t get_data_size()
        size_t get_default_data_size() except +translate_exception
        size_t get_data_alignment()
        size_t get_arrmeta_size()
        _type get_canonical_type()

        bint operator==(_type&)
        bint operator!=(_type&)

        bint match(_type&) except +translate_exception

        type_id_t get_type_id() const

cdef extern from "numpy_interop.hpp" namespace "pydynd":
    object numpy_dtype_obj_from__type(_type&) except +translate_exception

cdef extern from "dynd/types/type_alignment.hpp" namespace "dynd":
    _type dynd_make_unaligned_type "dynd::ndt::make_unaligned" (_type&) except +translate_exception

cdef extern from "dynd/types/byteswap_type.hpp" namespace "dynd":
    _type dynd_make_byteswap_type "dynd::ndt::byteswap_type::make" (_type&) except +translate_exception
    _type dynd_make_byteswap_type "dynd::ndt::byteswap_type::make" (_type&, _type&) except +translate_exception

cdef extern from 'dynd/types/fixed_bytes_type.hpp' namespace 'dynd':
    _type dynd_make_fixed_bytes_type 'dynd::ndt::make_fixed_bytes'(intptr_t, intptr_t) except +translate_exception

cdef extern from "dynd/types/fixed_dim_kind_type.hpp" namespace "dynd":
    _type dynd_make_fixed_dim_kind_type "dynd::ndt::fixed_dim_kind_type::make" (_type&) except +translate_exception
    _type dynd_make_fixed_dim_kind_type "dynd::ndt::fixed_dim_kind_type::make" (_type&, intptr_t) except +translate_exception

cdef extern from "dynd/types/var_dim_type.hpp" namespace "dynd":
    _type dynd_make_var_dim_type "dynd::ndt::var_dim_type::make" (_type&) except +translate_exception

cdef extern from 'dynd/types/datashape_formatter.hpp' namespace 'dynd':
    string dynd_format_datashape 'dynd::format_datashape' (_type&) except +translate_exception

cdef extern from "dynd/types/categorical_type.hpp" namespace "dynd":
    _type dynd_make_categorical_type "dynd::ndt::categorical_type::make" (_array&) except +translate_exception
    _type dynd_factor_categorical_type "dynd::ndt::factor_categorical" (_array&) except +translate_exception

cdef extern from 'dynd/types/callable_type.hpp':
    _type make_callable 'dynd::ndt::callable_type::make'(_type &, _array &)
    _type make_callable 'dynd::ndt::callable_type::make'(_type &, _type &, _type &)

cdef extern from 'dynd/types/tuple_type.hpp':
    _type _tuple 'dynd::ndt::tuple_type::make'() \
        except +translate_exception
    _type _tuple 'dynd::ndt::tuple_type::make'(const _array &) \
        except +translate_exception

cdef extern from 'dynd/types/struct_type.hpp':
    _type _struct 'dynd::ndt::struct_type::make'() \
        except +translate_exception
    _type _struct 'dynd::ndt::struct_type::make'(const _array &, const _array &) \
        except +translate_exception

cdef extern from 'gfunc_callable_functions.hpp' namespace 'pydynd':
    void add__type_names_to_dir_dict(_type&, object) except +translate_exception
    object get__type_dynamic_property(_type&, object) except +translate_exception

    # Function properties
    cdef cppclass _type_callable_wrapper:
        pass
    object _type_callable_call(_type_callable_wrapper&, object, object) except +translate_exception

    void init_w__type_callable_typeobject(object)

cdef extern from 'type_functions.hpp' namespace 'pydynd':
    void init_w_type_typeobject(object)

    _type make__type_from_pyobject(object) except +translate_exception

    object _type_get_shape(_type&) except +translate_exception
    object _type_get_kind(_type&) except +translate_exception
    object _type_get_type_id(_type&) except +translate_exception
    string _type_str(_type &)
    string _type_repr(_type &)

    _type dynd_make_convert_type(_type&, _type&) except +translate_exception
    _type dynd_make_view_type(_type&, _type&) except +translate_exception
    _type dynd_make_fixed_string_type(int, object) except +translate_exception
    _type dynd_make_string_type(object) except +translate_exception
    _type dynd_make_pointer_type(_type&) except +translate_exception
    _type dynd_make_struct_type(object, object) except +translate_exception
    _type dynd_make_cstruct_type(object, object) except +translate_exception
    _type dynd_make_fixed_dim_type(object, _type&) except +translate_exception
    _type dynd_make_cfixed_dim_type(object, _type&, object) except +translate_exception

cdef class type(object):
    cdef _type v
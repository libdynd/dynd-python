cdef extern from 'dynd/types/type_id.hpp' namespace 'dynd':
    ctypedef enum type_id_t:
        uninitializedtype_id
        booltype_id
        int8type_id
        int16type_id
        int32type_id
        int64type_id
        int128type_id
        uint8type_id
        uint16type_id
        uint32type_id
        uint64type_id
        uint128type_id
        float16type_id
        float32type_id
        float64type_id
        float128type_id
        complex_float32type_id
        complex_float64type_id
        voidtype_id

        pointertype_id
        void_pointertype_id

        bytestype_id
        fixed_bytestype_id

        chartype_id
        stringtype_id
        fixed_stringtype_id

        categoricaltype_id

        optiontype_id

        datetype_id
        timetype_id
        datetimetype_id
        busdatetype_id

        jsontype_id

        tupletype_id
        structtype_id

        fixed_dimtype_id
        offset_dimtype_id
        var_dimtype_id

        callabletype_id

        typetype_id

cdef extern from 'dynd/type.hpp' namespace 'dynd::ndt':
    cdef cppclass type 'dynd::ndt::type':
        type()
        type(type_id_t) except +translate_exception
        type(string&) except +translate_exception

        size_t get_data_size()
        size_t get_default_data_size() except +translate_exception
        size_t get_data_alignment()
        size_t get_arrmeta_size()
        type get_canonicaltype()

        bint operator==(type&)
        bint operator!=(type&)

        bint match(type&) except +translate_exception

        type_id_t gettype_id() const

cdef extern from "dynd/types/type_alignment.hpp" namespace "dynd":
    type dynd_make_unalignedtype "dynd::ndt::make_unaligned" (type&) except +translate_exception

cdef extern from "dynd/types/byteswaptype.hpp" namespace "dynd":
    type dynd_make_byteswaptype "dynd::ndt::byteswaptype::make" (type&) except +translate_exception
    type dynd_make_byteswaptype "dynd::ndt::byteswaptype::make" (type&, type&) except +translate_exception

cdef extern from 'dynd/types/fixed_bytestype.hpp' namespace 'dynd':
    type dynd_make_fixed_bytestype 'dynd::ndt::make_fixed_bytes'(intptr_t, intptr_t) except +translate_exception

cdef extern from "dynd/types/fixed_dim_kindtype.hpp" namespace "dynd":
    type dynd_make_fixed_dim_kindtype "dynd::ndt::fixed_dim_kindtype::make" (type&) except +translate_exception
    type dynd_make_fixed_dim_kindtype "dynd::ndt::fixed_dim_kindtype::make" (type&, intptr_t) except +translate_exception

cdef extern from "dynd/types/var_dimtype.hpp" namespace "dynd":
    type dynd_make_var_dimtype "dynd::ndt::var_dimtype::make" (type&) except +translate_exception

cdef extern from 'dynd/types/datashape_formatter.hpp' namespace 'dynd':
    string dynd_format_datashape 'dynd::format_datashape' (type&) except +translate_exception

cdef extern from "dynd/types/categoricaltype.hpp" namespace "dynd":
    type dynd_make_categoricaltype "dynd::ndt::categoricaltype::make" (array&) except +translate_exception
    type dynd_factor_categoricaltype "dynd::ndt::factor_categorical" (array&) except +translate_exception

cdef extern from 'dynd/types/callabletype.hpp':
    type make_callable 'dynd::ndt::callabletype::make'(type &, array &)
    type make_callable 'dynd::ndt::callabletype::make'(type &, type &, type &)

cdef extern from 'dynd/types/tupletype.hpp':
    type _tuple 'dynd::ndt::tupletype::make'() \
        except +translate_exception
    type _tuple 'dynd::ndt::tupletype::make'(const array &) \
        except +translate_exception

cdef extern from 'dynd/types/structtype.hpp':
    type _struct 'dynd::ndt::structtype::make'() \
        except +translate_exception
    type _struct 'dynd::ndt::structtype::make'(const array &, const array &) \
        except +translate_exception

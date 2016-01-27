cdef extern from 'dynd/types/type_id.hpp' namespace 'dynd':
    ctypedef enum type_id_t:
        uninitialized_id
        bool_id
        int8_id
        int16_id
        int32_id
        int64_id
        int128_id
        uint8_id
        uint16_id
        uint32_id
        uint64_id
        uint128_id
        float16_id
        float32_id
        float64_id
        float128_id
        complex_float32_id
        complex_float64_id
        void_id

        pointer_id
        void_pointer_id

        bytes_id
        fixed_bytes_id

        char_id
        string_id
        fixed_string_id

        categorical_id

        option_id

        date_id
        time_id
        datetime_id
        busdate_id

        json_id

        tuple_id
        struct_id

        fixed_dim_id
        offset_dim_id
        var_dim_id

        callable_id

        type_id

        uint_kind_id
        int_kind_id
        bool_kind_id
        float_kind_id
        complex_kind_id

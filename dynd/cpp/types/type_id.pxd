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

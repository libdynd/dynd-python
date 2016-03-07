from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.pair cimport pair

from .types.type_id cimport *

from ..config cimport translate_exception

cdef extern from 'dynd/type.hpp' namespace 'dynd::ndt' nogil:
    cdef cppclass type
    ctypedef pair[type, const char *] property_t

    cdef cppclass type:
        type()
        type(type_id_t) except +translate_exception
        type(string&) except +translate_exception

        size_t get_data_size()
        size_t get_default_data_size() except +translate_exception
        size_t get_data_alignment()
        size_t get_arrmeta_size()
        type get_canonical_type()

        map[string, property_t] get_properties()

        bool is_builtin()
        bool is_null()

        bint operator==(type&)
        bint operator!=(type&)

        bint match(type&) except +translate_exception

        type_id_t get_id() const
        type_id_t get_base_id() const

    type make_type[T]() except +translate_exception


from libcpp.string cimport string

from .types.type_id cimport type_id_t

from ..config cimport translate_exception
from .array cimport array

cdef extern from 'dynd/type.hpp' namespace 'dynd::ndt' nogil:
    cdef cppclass type:
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

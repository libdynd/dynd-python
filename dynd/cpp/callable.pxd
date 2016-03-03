from libc.string cimport const_char
from libcpp.pair cimport pair
from libcpp cimport bool

from ..config cimport translate_exception
from .array cimport array
from .type cimport type

ctypedef const_char* const_charptr

cdef extern from 'dynd/callable.hpp' namespace 'dynd::nd' nogil:
    cdef cppclass callable:
        callable()

        type get_array_type()

        bool is_null()

        array call(size_t, array *, size_t, pair[const_charptr, array] *) except +translate_exception
        array operator()(...) except +translate_exception

    callable make_callable 'dynd::nd::callable::make'[T](...) except +translate_exception


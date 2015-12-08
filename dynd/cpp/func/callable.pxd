from libcpp cimport bool

from ..array cimport array
from ..type cimport type

cdef extern from 'dynd/func/callable.hpp' namespace 'dynd::nd' nogil:
    cdef cppclass callable:
        callable()

        type get_array_type()

        bool is_null()

        array operator()(...)

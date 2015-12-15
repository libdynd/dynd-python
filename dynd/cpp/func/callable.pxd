from libcpp.pair cimport pair
from libcpp cimport bool

from ...config cimport translate_exception
from ..array cimport array
from ..type cimport type

cdef extern from 'dynd/callable.hpp' namespace 'dynd::nd' nogil:
    cdef cppclass callable:
        callable()

        type get_array_type()

        bool is_null()

        array call(size_t, array *, size_t, pair[char *, array] *)
        array operator()(...)

    callable apply[T](T) except +translate_exception

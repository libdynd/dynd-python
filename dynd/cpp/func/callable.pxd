from ..type cimport type

cdef extern from 'dynd/func/callable.hpp' namespace 'dynd::nd':
    cdef cppclass callable:
        callable()

        type get_array_type()

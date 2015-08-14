from ..config cimport translate_exception
from libc.stdint cimport intptr_t
from .type cimport type

cdef extern from 'dynd/array.hpp' namespace 'dynd::nd' nogil:
    cdef cppclass array:

        array()
        array(type *, int)

        type get_type()
        type get_dtype()
        type get_dtype(size_t)
        intptr_t get_ndim()
        intptr_t get_dim_size() except +translate_exception

        array view_scalars(type&) except +translate_exception

        array operator<(array &)
        array operator<=(array &)
        array operator==(array &)
        array operator!=(array &)
        array operator>=(array &)
        array operator>(array &)

    array groupby(array&, array&, type) except +translate_exception
    array groupby(array&, array&) except +translate_exception

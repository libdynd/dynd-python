from ..config cimport translate_exception
from .type cimport type

cdef extern from 'dynd/array.hpp' namespace 'dynd::nd':
    cdef cppclass array:

        array()
        array(type *, int)

        type get_type()
        type get_dtype()
        type get_dtype(size_t)
        intptr_t get_ndim()
        intptr_t get_dim_size() except +translate_exception

        array view_scalars(type&) except +translate_exception

    array operator<(const array &, const array &)
    array operator<=(const array &, const array &)
    array operator==(const array &, const array &)
    array operator!=(const array &, const array &)
    array operator>=(const array &, const array &)
    array operator>(const array &, const array &)

    array dynd_groupby(array&, array&, type) except +translate_exception
    array dynd_groupby(array&, array&) except +translate_exception

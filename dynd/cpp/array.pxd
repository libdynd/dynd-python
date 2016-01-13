from ..config cimport translate_exception
from libc.stdint cimport intptr_t
from libcpp cimport bool
from .type cimport type

cdef extern from 'dynd/array.hpp' namespace 'dynd::nd' nogil:

    cdef enum:
        read_access_flag
        write_access_flag
        immutable_access_flag
        readwrite_access_flags
        default_access_flags

    cdef cppclass array:

        array()
        array(type *, int)

        type get_type()
        type get_dtype()
        type get_dtype(size_t)
        intptr_t get_ndim()
        intptr_t get_dim_size() except +translate_exception

        array view_scalars(type&) except +translate_exception

        bool is_null()

        array operator<(array &)
        array operator<=(array &)
        array operator==(array &)
        array operator!=(array &)
        array operator>=(array &)
        array operator>(array &)

        void assign(array &) except +translate_exception

    array empty(type &tp) except +translate_exception

    # These should be usable with Cython's operator overloading syntax, but
    # the exception handling doesn't work for overloaded operators in Cython
    # versions before 0.24, so, for now, this will have to do.
    array array_add "operator+"(array&, array&) except +translate_exception
    array array_subtract "operator-"(array&, array&) except +translate_exception
    array array_multiply "operator*"(array&, array&) except +translate_exception
    array array_divide "operator/"(array&, array&) except +translate_exception

    array groupby(array&, array&, type) except +translate_exception
    array groupby(array&, array&) except +translate_exception

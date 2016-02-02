from ..type cimport type
from ..array cimport array
from libcpp.vector cimport vector

cdef extern from 'dynd/types/callable_type.hpp':
    type make_callable 'dynd::ndt::callable_type::make'(type &, vector[type] &)
    type make_callable 'dynd::ndt::callable_type::make'(type &, type &, type &)

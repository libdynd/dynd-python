from ..type cimport type
from ..array cimport array

cdef extern from 'dynd/types/callable_type.hpp':
    type make_callable 'dynd::ndt::callable_type::make'(type &, array &)
    type make_callable 'dynd::ndt::callable_type::make'(type &, type &, type &)

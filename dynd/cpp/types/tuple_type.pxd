from libcpp.vector cimport vector

from ..type cimport type
from ..array cimport array

from ...config cimport translate_exception

cdef extern from 'dynd/types/tuple_type.hpp':
    type make_tuple 'dynd::ndt::tuple_type::make'() \
        except +translate_exception
    type make_tuple 'dynd::ndt::tuple_type::make'(const vector[type] &) \
        except +translate_exception

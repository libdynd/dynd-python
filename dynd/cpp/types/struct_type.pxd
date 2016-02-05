from libcpp.string cimport string
from libcpp.vector cimport vector

from ..array cimport array
from ..type cimport type

from ...config cimport translate_exception

cdef extern from 'dynd/types/struct_type.hpp':
    type make_struct 'dynd::ndt::struct_type::make'() \
        except +translate_exception
    type make_struct 'dynd::ndt::struct_type::make'(const vector[string] &, const vector[type] &) \
        except +translate_exception

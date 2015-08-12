from ..array cimport array
from ..type cimport type

from ...config cimport translate_exception

cdef extern from 'dynd/types/struct_type.hpp':
    type make_struct 'dynd::ndt::struct_type::make'() \
        except +translate_exception
    type make_struct 'dynd::ndt::struct_type::make'(const array &, const array &) \
        except +translate_exception

from libcpp.string cimport string

from ..type cimport type
from ..array cimport array

from ...config cimport translate_exception

cdef extern from 'dynd/types/datashape_formatter.hpp' namespace 'dynd':
    string format_datashape(type&) except +translate_exception
    string format_datashape(array&) except +translate_exception

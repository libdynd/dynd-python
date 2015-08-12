from libcpp.string cimport string

from ..type cimport type

from ...config cimport translate_exception

cdef extern from 'dynd/types/datashape_formatter.hpp' namespace 'dynd':
    string dynd_format_datashape 'dynd::format_datashape' (type&) except +translate_exception

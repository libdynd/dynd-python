from libcpp.string cimport string

from ..config cimport translate_exception
from .type cimport type
from .array cimport array

cdef extern from 'dynd/json_parser.hpp' namespace 'dynd::ndt::json':
    type discover(string) except +translate_exception

cdef extern from 'dynd/json_parser.hpp' namespace 'dynd::nd::json':
    array parse(type, string) except +translate_exception

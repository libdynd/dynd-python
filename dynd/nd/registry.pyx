# cython: c_string_type=str, c_string_encoding=ascii

from ..cpp.func.callable cimport callable
from ..cpp.func.callable_registry cimport callable_registry

from ..config cimport translate_exception
from .callable cimport dynd_nd_callable_from_cpp

cdef extern from 'assign.hpp':
    void assign_init() except +translate_exception

assign_init()

def get_published_callables():
    for pair in callable_registry:
        yield pair.first, dynd_nd_callable_from_cpp(pair.second)

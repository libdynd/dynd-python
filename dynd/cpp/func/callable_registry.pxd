from libcpp.map cimport map
from libcpp.string cimport string

from .callable cimport callable

from ...config cimport translate_exception

cdef extern from "dynd/callable_registry.hpp" namespace "dynd::nd" nogil:
    cdef cppclass callable_registry:
        map[string, callable] &get_regfunctions() except +translate_exception

    cdef extern callable_registry callable_registry

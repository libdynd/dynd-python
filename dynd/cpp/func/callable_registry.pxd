from libcpp.map cimport map
from libcpp.string cimport string

from .callable cimport callable

from ...config cimport translate_exception

cdef extern from "dynd/func/callable_registry.hpp" namespace "dynd::func" nogil:
    map[string, callable] &get_regfunctions() except +translate_exception

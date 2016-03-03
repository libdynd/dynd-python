from libcpp.map cimport map
from libcpp.string cimport string

from .callable cimport callable

from ..config cimport translate_exception

cdef extern from "dynd/callable_registry.hpp" namespace "dynd::nd" nogil:
    cdef cppclass callable_registry:
        cppclass iterator:
            pass

        callable operator[](string)

        map[string, callable].iterator find(string)

        map[string, callable].iterator begin()
        map[string, callable].const_iterator cbegin()

        map[string, callable].iterator end()
        map[string, callable].const_iterator cend()

    cdef extern callable_registry callable_registry

# cython: c_string_type=str, c_string_encoding=ascii

from cython.operator cimport dereference, postincrement

from libcpp.map cimport map
from libcpp.string cimport string

from ..cpp.func.callable cimport callable
from ..cpp.func.callable_registry cimport callable_registry

from ..config cimport translate_exception
from .callable cimport dynd_nd_callable_from_cpp

cdef extern from 'assign.hpp':
    void assign_init() except +translate_exception

assign_init()

cdef extern from *:
    bint is_py_2 "(PY_MAJOR_VERSION == 2)"

def get_published_callables():
    py_reg = dict()
    for pair in callable_registry:
        py_reg[pair.first] = dynd_nd_callable_from_cpp(pair.second)

    return py_reg

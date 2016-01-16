from cython.operator cimport dereference, postincrement

from libcpp.map cimport map
from libcpp.string cimport string

from ..cpp.func.callable cimport callable
from ..cpp.func.callable_registry cimport get_regfunctions

from ..config cimport translate_exception
from .callable cimport dynd_nd_callable_from_cpp

cdef extern from 'assign.hpp':
    void assign_init() except +translate_exception

assign_init()

cdef extern from *:
    bint is_py_2 "(PY_MAJOR_VERSION == 2)"

def get_published_callables():
    py_reg = dict()
    cdef map[string, callable].const_iterator it = get_regfunctions().const_begin()
    cdef map[string, callable].const_iterator end = get_regfunctions().const_end()
    while it != end:
        if is_py_2:
            py_reg[dereference(it).first] = dynd_nd_callable_from_cpp(dereference(it).second)
        else:
            key = dereference(it).first
            py_reg[key.decode('UTF-8')] = dynd_nd_callable_from_cpp(dereference(it).second)
        postincrement(it)
    return py_reg

# cython: c_string_type=str, c_string_encoding=ascii

from ..cpp.func.callable cimport callable
from ..cpp.func.callable_registry cimport callable_registry

from ..config cimport translate_exception
from .callable cimport dynd_nd_callable_from_cpp
from .array cimport _registry_assign_init as assign_init

assign_init()

def get_published_callables():
    for pair in callable_registry:
        yield pair.first, dynd_nd_callable_from_cpp(pair.second)

# cython: c_string_type=str, c_string_encoding=ascii

from ..cpp.callable cimport callable
from ..cpp.callable cimport callables

from ..config cimport translate_exception
from .callable cimport wrap
from .array cimport _registry_assign_init as assign_init

assign_init()

def get_published_callables():
    for f in callables():
        yield f.first, wrap(f.second)

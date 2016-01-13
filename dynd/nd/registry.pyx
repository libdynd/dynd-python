from ..cpp.func.registry cimport get_published_callables as _get_published_callables

from ..config cimport translate_exception

cdef extern from 'assign.hpp':
    void _assign_init 'assign_init'() except +translate_exception

_assign_init()

def get_published_callables():
  return _get_published_callables()

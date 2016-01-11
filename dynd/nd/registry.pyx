from ..cpp.func.registry cimport get_published_callables as _get_published_callables

from ..config cimport translate_exception

cdef extern from 'copy_from_pyobject_arrfunc.hpp':
    void _add_overloads 'add_overloads'() except +translate_exception

def add_overloads():
    _add_overloads()

def get_published_callables():
  return _get_published_callables()

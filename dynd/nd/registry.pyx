# cython: c_string_type=str, c_string_encoding=ascii

import imp
import sys

from cython.operator cimport dereference

from ..cpp.callable cimport callable, get, reg_entry, observe
from ..cpp.config cimport load

from ..config cimport translate_exception
from .callable cimport wrap
from .array cimport _registry_assign_init as assign_init

assign_init()

cdef _publish_callables(mod, reg_entry entry):
    for pair in entry:
        if (pair.second.is_namespace()):
            new_mod = imp.new_module(pair.first)
            _publish_callables(new_mod, pair.second)
            setattr(mod, pair.first, new_mod)
        else:
            setattr(mod, pair.first, wrap(pair.second.value()))

def publish_callables(mod):
    return _publish_callables(mod, get())

cdef void observer(const char *name, reg_entry *entry) with gil:
    from . import __name__
    mod = sys.modules[__name__]

    if entry.is_namespace():
        _publish_callables(mod, dereference(entry))

observe(&observer)

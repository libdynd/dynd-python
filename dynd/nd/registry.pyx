# cython: c_string_type=str, c_string_encoding=ascii

import imp
import sys

from cython.operator cimport dereference

from ..config cimport translate_exception
from .callable cimport wrap
from .array cimport _registry_assign_init as assign_init

assign_init()

from cython.operator import preincrement

cdef void update(registry_entry *parent_entry, const char *name, registry_entry *entry) with gil:
    if (entry.is_namespace()):
        try:
            new_mod = sys.modules[entry.path()]
        except KeyError:
            new_mod = imp.new_module(entry.path())
            sys.modules[entry.path()] = new_mod
        propagate(entry)
        if (not parent_entry.path().empty()):
            mod = sys.modules[parent_entry.path()]
            setattr(mod, name, new_mod)
    else:
        mod = sys.modules[parent_entry.path()]
        setattr(mod, name, wrap(entry.value()))

cdef void propagate(registry_entry *entry):
    entry.observe(update)

    cdef map[string, registry_entry].iterator it = entry.begin()
    while (it != entry.end()):
        update(entry, dereference(it).first.c_str(), &dereference(it).second)
        preincrement(it)

def propagate_all():
    return propagate(&registered())

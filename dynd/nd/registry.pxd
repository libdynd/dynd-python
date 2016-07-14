from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.string cimport string

from ..cpp.callable cimport callable
from ..cpp.registry cimport registered, registry_entry

cdef extern void propagate(registry_entry *entry)

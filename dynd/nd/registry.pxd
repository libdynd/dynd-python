
from ..cpp.registry cimport registered, registry_entry

cdef extern _publish_callables(registry_entry &entry)

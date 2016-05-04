from .cpp.config cimport load as _load

cdef:
    void translate_exception()

cdef api object DyND_PyExc_BroadcastError

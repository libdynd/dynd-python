from ..cpp.func.callable cimport callable as _callable

cdef api class callable(object)[object dynd_nd_callable_pywrapper, type dynd_nd_callable_pywrapper_type]:
    cdef _callable v

cdef api _callable callable_to_cpp(callable) except *
cdef api callable callable_from_cpp(_callable)

from ..cpp.callable cimport callable as _callable

cdef api class callable(object)[object dynd_nd_callable_pywrapper,
                                type dynd_nd_callable_pywrapper_type]:
    cdef _callable v

cdef api _callable dynd_nd_callable_to_cpp(callable) except *
# Provide an API for returning as a pointer since Cython can't handle
# returning references yet.
cdef api _callable *dynd_nd_callable_to_ptr(callable) except *
cdef api callable dynd_nd_callable_from_cpp(const _callable &)

from ..cpp.func.callable cimport callable as _callable

cdef class callable(object):
    cdef _callable v
    cdef inline _callable as_cpp(callable self) except *:
        if self is None:
            raise TypeError("Cannot get c++ callable from None.")
        return self.v
    @staticmethod
    cdef callable from_cpp(_callable v)

from ..cpp.array cimport array as _array
from ..cpp.func.callable cimport callable as _callable

cdef api class array(object)[object dynd_nd_array_pywrapper,
                             type dynd_nd_array_pywrapper_type]:
    cdef _array v

cdef _array as_cpp_array(object obj) except *
cpdef array asarray(object obj)

cdef api _array dynd_nd_array_to_cpp(array) except *
cdef api array dynd_nd_array_from_cpp(_array)

cdef extern from 'assign.hpp':
    cdef cppclass declfunc:
        _callable get()

    cdef extern declfunc assign_pyarrayscalarobject

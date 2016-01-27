from ..cpp.array cimport array as _array

cdef api class array(object)[object dynd_nd_array_pywrapper,
                             type dynd_nd_array_pywrapper_type]:
    cdef _array v

cdef _array as_cpp_array(object obj) except *
cpdef array asarray(object obj)

cdef api _array dynd_nd_array_to_cpp(array) except *
cdef api _array *dynd_nd_array_to_ptr(array) except *
cdef api array dynd_nd_array_from_cpp(_array)

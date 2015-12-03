from ..cpp.array cimport array as _array

cdef api class array(object)[object dynd_nd_array_pywrapper, type dynd_nd_array_pywrapper_type]:
    cdef _array v

cpdef array asarray(obj, access=*)

cdef api _array dynd_nd_array_to_cpp(array) except *
cdef api array dynd_nd_array_from_cpp(_array)

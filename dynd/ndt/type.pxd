from ..cpp.type cimport type as _type

cdef api class type(object)[object dynd_ndt_type_pywrapper,
                            type dynd_ndt_type_pywrapper_type]:
    cdef _type v

cdef api _type dynd_ndt_type_to_cpp(type) except *
cdef api _type *dynd_ndt_type_to_ptr(type) except *
cdef api type dynd_ndt_type_from_cpp(const _type&)
cdef api _type as_cpp_type(object o) except *
cpdef type astype(object o)

cdef object as_numba_type(_type)
cdef _type from_numba_type(object)
cdef api _type cpp_type_for(object) except *

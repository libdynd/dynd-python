from ..cpp.type cimport type as _type
from cpython.object cimport PyObject, PyTypeObject

cdef api class type(object)[object dynd_ndt_type_pywrapper,
                            type dynd_ndt_type_pywrapper_type]:
    cdef _type v

cdef api _type dynd_ndt_type_to_cpp(type) nogil except *
cdef api _type *dynd_ndt_type_to_ptr(type) nogil except *
cdef api type wrap(const _type&)
cdef api _type as_cpp_type(object o) except *
cpdef type astype(object o)

cdef object as_numba_type(_type)
cdef _type from_numba_type(object)
cdef api _type cpp_type_for(object) except *

cdef void _register_nd_array_type_deduction(PyTypeObject *array_type, _type (*get_type)(PyObject *))

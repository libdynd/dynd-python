from cpython.object cimport PyObject

from ..cpp.array cimport array as _array
from ..cpp.callable cimport callable as _callable
from ..cpp.type cimport type as _type

cdef api class array(object)[object dynd_nd_array_pywrapper,
                             type dynd_nd_array_pywrapper_type]:
    cdef _array v

cdef _array as_cpp_array(object obj) except *
cpdef array asarray(object obj)

cdef api _array dynd_nd_array_to_cpp(array) except *
cdef api _array *dynd_nd_array_to_ptr(array) except *
cdef api array dynd_nd_array_from_cpp(_array)

cdef _callable _functional_apply(_type t, object o) except *
cdef void _registry_assign_init() except *
cdef object _numpy_dtype_from__type(const _type &tp)
cdef bint _is_numpy_dtype(PyObject *o) except *
cdef _type _xtype_for_prefix(object o) except *

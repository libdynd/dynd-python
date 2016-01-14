from libc.stdint cimport intptr_t

from ..cpp.func.callable cimport callable as _callable
from ..cpp.type cimport type as _type

from ..config cimport translate_exception

cdef extern from 'functional.hpp':
    _callable _apply 'apply'(_type, object) except +translate_exception

cdef extern from "kernels/apply_jit_kernel.hpp" namespace "pydynd::nd::functional":
    _callable _apply_jit "pydynd::nd::functional::apply_jit"(const _type &tp, intptr_t)

    cdef cppclass jit_dispatcher:
        jit_dispatcher(object, object (*)(object, intptr_t, const _type *))

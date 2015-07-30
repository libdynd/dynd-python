from libc.stdint cimport intptr_t

from ..config cimport translate_exception
from dynd.ndt.type cimport _type
from .callable cimport _callable

from cpython.ref cimport PyObject

cdef extern from 'dynd/func/elwise.hpp' namespace 'dynd::nd::functional':
    _callable _elwise 'dynd::nd::functional::elwise'(_callable) except +translate_exception

cdef extern from "arrfunc_from_pyfunc.hpp" namespace "pydynd::nd::functional":
    _callable _apply 'pydynd::nd::functional::apply'(object, object) except +translate_exception

cdef extern from "kernels/apply_jit_kernel.hpp" namespace "pydynd::nd::functional":
    _callable _apply_jit "pydynd::nd::functional::apply_jit"(const _type &tp, intptr_t)

    cdef cppclass jit_dispatcher:
        jit_dispatcher(object, object (*)(object, intptr_t, const _type *))

cdef extern from 'dynd/func/multidispatch.hpp' namespace 'dynd::nd::functional':
    _callable _multidispatch 'dynd::nd::functional::multidispatch'[T](_type, T, size_t) \
        except +translate_exception
    _callable _multidispatch 'dynd::nd::functional::multidispatch'[T](_type, T, T) \
        except +translate_exception
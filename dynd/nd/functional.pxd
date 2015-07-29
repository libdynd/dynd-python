from libc.stdint cimport intptr_t

from ..config cimport translate_exception
from dynd.ndt.type cimport _type
from .callable cimport _callable

from cpython.ref cimport PyObject

cdef extern from 'dynd/func/elwise.hpp' namespace 'dynd::nd::functional':
    _callable _elwise 'dynd::nd::functional::elwise'(_callable) except +translate_exception

cdef extern from "arrfunc_from_pyfunc.hpp" namespace "pydynd::nd::functional":
    _callable _apply 'pydynd::nd::functional::apply'(object, object) except +translate_exception

cdef extern from "kernels/apply_numba_kernel.hpp" namespace "pydynd::nd::functional":
    _callable numba_helper "pydynd::nd::functional::numba_helper"(const _type &tp, intptr_t)

    _callable &dereference "pydynd::nd::functional::dereference"(_callable *)

    cdef cppclass jit_dispatcher:
        jit_dispatcher(object, object (*)(object func,
            const _type &dst_type, intptr_t nsrc, const _type *src_tp))

cdef extern from 'dynd/func/multidispatch.hpp' namespace 'dynd::nd::functional':
    _callable _multidispatch2 'dynd::nd::functional::multidispatch'[T](_type, T, size_t) \
        except +translate_exception
    _callable _multidispatch 'dynd::nd::functional::multidispatch'[T](_type, T, T) \
        except +translate_exception

cdef extern from '<functional>' namespace 'std':
    cdef cppclass reference_wrapper[T]:
        pass

    reference_wrapper[T] ref[T](T &)
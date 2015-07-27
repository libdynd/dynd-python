from ..config cimport translate_exception
from dynd.ndt.type cimport _type
from .callable cimport _callable

cdef extern from 'dynd/func/elwise.hpp' namespace 'dynd::nd::functional':
    _callable _elwise 'dynd::nd::functional::elwise'(_callable) except +translate_exception

cdef extern from "arrfunc_from_pyfunc.hpp" namespace "pydynd::nd::functional":
    _callable _apply 'pydynd::nd::functional::apply'(object, object) except +translate_exception

cdef extern from 'dynd/func/multidispatch.hpp' namespace 'dynd::nd::functional':
    _callable _multidispatch 'dynd::nd::functional::multidispatch'[T](_type, T, T) except +translate_exception
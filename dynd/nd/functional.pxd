from ..config cimport translate_exception
from .arrfunc cimport _arrfunc

cdef extern from 'dynd/func/elwise.hpp' namespace 'dynd::nd::functional':
    _arrfunc _elwise 'dynd::nd::functional::elwise'(_arrfunc) except +translate_exception

cdef extern from "arrfunc_from_pyfunc.hpp" namespace "pydynd::nd::functional":
    _arrfunc _apply 'pydynd::nd::functional::apply'(object, object) except +translate_exception

cdef extern from "arrfunc_functions.hpp" namespace "pydynd":
    object wrap_array(const _arrfunc &af)
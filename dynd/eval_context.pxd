from config cimport translate_exception

cdef extern from "dynd/eval/eval_context.hpp" namespace "dynd":
    cdef cppclass _eval_context "dynd::eval::eval_context":
        pass

cdef extern from "eval_context_functions.hpp" namespace "pydynd":
    void init_w_eval_context_typeobject(object)

    _eval_context *new_eval_context(object) except +translate_exception
    void dynd_modify_default_eval_context "pydynd::modify_default_eval_context" (object) except +translate_exception
    object get_eval_context_errmode(object) except +translate_exception
    object get_eval_context_cuda_device_errmode(object) except +translate_exception
    object get_eval_context_date_parse_order(object) except +translate_exception
    object get_eval_context_century_window(object) except +translate_exception
    object get_eval_context_repr(object) except +translate_exception

cdef class eval_context:
    # NOTE: This layout is also accessed from C++
    cdef _eval_context *ectx
    cdef bint own_ectx
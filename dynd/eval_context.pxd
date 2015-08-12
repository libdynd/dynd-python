from .cpp.eval.eval_context cimport eval_context as _eval_context

cdef class eval_context:
    # NOTE: This layout is also accessed from C++
    cdef _eval_context *ectx
    cdef bint own_ectx

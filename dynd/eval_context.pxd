from config cimport translate_exception

cdef class eval_context:
    # NOTE: This layout is also accessed from C++
    cdef _eval_context *ectx
    cdef bint own_ectx

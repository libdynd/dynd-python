#
# Copyright (C) 2011-15 DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

from translate_except cimport translate_exception

cdef extern from "dynd/types/date_util.hpp" namespace "dynd":
    cdef enum date_parse_order_t:
        date_parse_no_ambig
        date_parse_ymd
        date_parse_mdy
        date_parse_dmy

cdef extern from "dynd/eval/eval_context.hpp" namespace "dynd":
    cdef cppclass eval_context "dynd::eval::eval_context":
        pass

cdef extern from "eval_context_functions.hpp" namespace "pydynd":
    void init_w_eval_context_typeobject(object)

    eval_context *new_eval_context(object) except +translate_exception
    void dynd_modify_default_eval_context "pydynd::modify_default_eval_context" (object) except +translate_exception
    object get_eval_context_errmode(object) except +translate_exception
    object get_eval_context_cuda_device_errmode(object) except +translate_exception
    object get_eval_context_date_parse_order(object) except +translate_exception
    object get_eval_context_century_window(object) except +translate_exception
    object get_eval_context_repr(object) except +translate_exception

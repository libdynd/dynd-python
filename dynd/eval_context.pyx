from libc.stdint cimport uintptr_t

from .config cimport translate_exception

cdef extern from "eval_context_functions.hpp" namespace "pydynd":
    void init_w_eval_context_typeobject(object)

    _eval_context *new_eval_context(object) except +translate_exception
    void dynd_modify_default_eval_context "pydynd::modify_default_eval_context" (object) except +translate_exception
    object get_eval_context_errmode(object) except +translate_exception
    object get_eval_context_cuda_device_errmode(object) except +translate_exception
    object get_eval_context_date_parse_order(object) except +translate_exception
    object get_eval_context_century_window(object) except +translate_exception
    object get_eval_context_repr(object) except +translate_exception

cdef class eval_context(object):
    """
    nd.eval_context(reset=False,
                    errmode=None,
                    cuda_device_errmode=None,
                    date_parse_order=None,
                    century_window=None)

    Create a dynd evaluation context, overriding the defaults via
    the chosen parameters. Evaluation contexts can be used to
    adjust default settings

    Parameters
    ----------
    reset : bool, optional
        If set to true, first resets the evaluation context to
        factory settings instead of starting with the default
        evaluation context.
    errmode : 'inexact', 'fractional', 'overflow', 'nocheck', optional
        The default error mode used in computations when none is specified.
    cuda_device_errmode : 'inexact', 'fractional', 'overflow', 'nocheck', optional
        The default error mode used in cuda computations when none is
        specified.
    date_parse_order : 'NoAmbig', 'YMD', 'MDY', 'DMY', optional
        How to interpret dates being parsed when the order of year, month and
        day is ambiguous from the format.
    century_window : int, optional
        Whether and how to interpret two digit years. If 0, disallow them.
        If 1-99, use a sliding window beginning that number of years ago.
        If greater than 1000, use a fixed window starting at that year.
    """

    def __cinit__(self, *args, **kwargs):
        self.own_ectx = False
        if len(args) > 0:
            raise TypeError('nd.eval_context() accepts no positional args')

        # Start with a copy of the default eval context
        self.ectx = new_eval_context(kwargs)
        self.own_ectx = True

    def __dealloc__(self):
        if self.own_ectx:
            del self.ectx

    property errmode:
        def __get__(self):
            return get_eval_context_errmode(self)

    property cuda_device_errmode:
        def __get__(self):
            return get_eval_context_cuda_device_errmode(self)

    property date_parse_order:
        def __get__(self):
            return get_eval_context_date_parse_order(self)

    property century_window:
        def __get__(self):
            return get_eval_context_century_window(self)

    property _ectx_ptr:
        def __get__(self):
            return <uintptr_t>self.ectx

    def __str__(self):
        return get_eval_context_repr(self)

    def __repr__(self):
        return get_eval_context_repr(self)

init_w_eval_context_typeobject(eval_context)

def modify_default_eval_context(**kwargs):
    """
    nd.modify_default_eval_context(reset=False,
                    errmode=None,
                    cuda_device_errmode=None,
                    date_parse_order=None,
                    century_window=None)
    Modify the default dynd evaluation context, overriding the defaults via
    the chosen parameters. This is not recommended for typical use
    except in interactive sessions. Using the ``ectx=`` parameter to
    evaluation methods is preferred in library code.
    Parameters
    ----------
    reset : bool, optional
        If set to true, first resets the default evaluation context to
        factory settings.
    errmode : 'inexact', 'fractional', 'overflow', 'nocheck', optional
        The default error mode used in computations when none is specified.
    cuda_device_errmode : 'inexact', 'fractional', 'overflow', 'nocheck', optional
        The default error mode used in cuda computations when none is
        specified.
    date_parse_order : 'NoAmbig', 'YMD', 'MDY', 'DMY', optional
        How to interpret dates being parsed when the order of year, month and
        day is ambiguous from the format.
    century_window : int, optional
        Whether and how to interpret two digit years. If 0, disallow them.
        If 1-99, use a sliding window beginning that number of years ago.
        If greater than 1000, use a fixed window starting at that year.
    """
    dynd_modify_default_eval_context(kwargs)

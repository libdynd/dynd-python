from dynd.ndt import arrfunc, type

def apply(tp_or_func, func = None):
    def make(tp, func):
        return wrap_array(_apply(func, tp))

    if func is None:
        if isinstance(tp_or_func, type):
            return lambda func: make(tp_or_func, func)

        return make(arrfunc(tp_or_func), tp_or_func)

    return make(tp_or_func, func)
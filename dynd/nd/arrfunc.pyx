from .array cimport array

cdef class arrfunc(array):
    """
    nd.arrfunc(func, proto)

    This holds a dynd nd.arrfunc object, which represents a single typed
    function. The particular abstraction this represents is still being
    sorted out.

    The constructor creates an arrfunc out of a Python function.

    Parameters
    ----------
    func : callable
        A Python function or object that implements __call__.
    proto : ndt.type
        A funcproto describing the types for the resulting arrfunc.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> af = nd.arrfunc(lambda x, y: [x, y], '(int, int) -> {x:int, y:int}')
    >>>
    >>> af(1, 10)
    nd.array([1, 10], type="{x : int32, y : int32}")
    >>> af(1, "test")
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "config.pyx", line 1340, in config.w_arrfunc.__call__ (config.cxx:9774)
    ValueError: parameter 2 to arrfunc does not match, expected int32, received string
    """

    def __call__(self, *args, **kwds):
        # Handle the keyword-only arguments
        ectx = kwds.pop('ectx', None)
#        if kwds:
#            msg = "nd.arrfunc call got an unexpected keyword argument '%s'"
#            raise TypeError(msg % (kwds.keys()[0]))
        return arrfunc_call(self, args, kwds, ectx)

init_w_arrfunc_typeobject(arrfunc)
from dynd.wrapper cimport set_wrapper_type
from dynd.ndt.type cimport wrap_type

cdef class callable(object):
    """
    nd.callable(func, proto)

    This holds a dynd nd.callable object, which represents a single typed
    function. The particular abstraction this represents is still being
    sorted out.

    The constructor creates an callable out of a Python function.

    Parameters
    ----------
    func : callable
        A Python function or object that implements __call__.
    proto : ndt.type
        A funcproto describing the types for the resulting callable.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> af = nd.callable(lambda x, y: [x, y], '(int, int) -> {x:int, y:int}')
    >>>
    >>> af(1, 10)
    nd.array([1, 10], type="{x : int32, y : int32}")
    >>> af(1, "test")
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "config.pyx", line 1340, in config.callable.__call__ (config.cxx:9774)
    ValueError: parameter 2 to callable does not match, expected int32, received string
    """

    property type:
        def __get__(self):
            return wrap_type(self.v.get_array_type())

    def __call__(self, *args, **kwds):
        # Handle the keyword-only arguments
        ectx = kwds.pop('ectx', None)
#        if kwds:
#            msg = "nd.callable call got an unexpected keyword argument '%s'"
#            raise TypeError(msg % (kwds.keys()[0]))
        return callable_call(self, args, kwds, ectx)

init_w_callable_typeobject(callable)
set_wrapper_type[_callable](callable)
from libc.string cimport const_char
from libcpp.vector cimport vector
from libcpp.pair cimport pair

from ..cpp.array cimport array as _array

from cython.operator cimport dereference

from ..config cimport translate_exception
from ..cpp.callable cimport const_charptr, stringstream
from .array cimport as_cpp_array, dynd_nd_array_from_cpp

cdef extern from *:
    # Hack to allow compile-time resolution of the Python version.
    # This can be used inside if-statements which will, in turn, be
    # removed by the compiler's optimizer.
    bint is_py_2 "(PY_MAJOR_VERSION == 2)"

ctypedef pair[const_charptr, _array] char_array_pair

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
            return wrap(dereference(self.v).get_type())

    def __call__(callable self, *args, **kwargs):
        cdef size_t nargs = len(args), nkwargs = len(kwargs)
        cdef vector[_array] cpp_args
        cpp_args.reserve(nargs)
        cdef vector[char_array_pair] cpp_kwargs
        cpp_kwargs.reserve(nkwargs)
        for ar in args:
            cpp_args.push_back(as_cpp_array(ar))
        if is_py_2:
            for s, ar in kwargs.iteritems():
                cpp_kwargs.push_back(char_array_pair(
                    <const_char*>s, as_cpp_array(ar)))
        else:
            for s, ar in kwargs.items():
                s_tmp = s.encode('UTF-8')
                cpp_kwargs.push_back(char_array_pair(
                    <const_char*>s_tmp, as_cpp_array(ar)))
        a = dynd_nd_array_from_cpp(dynd_nd_callable_to_cpp(self).call(
                   nargs, cpp_args.data(), nkwargs, cpp_kwargs.data()))
        return a

    def __repr__(self):
        cdef stringstream ss
        ss << self.v

        return ss.str()

cdef _callable dynd_nd_callable_to_cpp(callable c) except *:
    # Once this becomes a method of the type wrapper class, this check and
    # its corresponding exception handler declaration are no longer necessary
    # since the self parameter is guaranteed to never be None.
    if c is None:
        raise TypeError("Cannot extract DyND C++ callable from None.")
    return c.v

cdef _callable *dynd_nd_callable_to_ptr(callable c) except *:
    # Once this becomes a method of the type wrapper class, this check and
    # its corresponding exception handler declaration are no longer necessary
    # since the self parameter is guaranteed to never be None.
    if c is None:
        raise TypeError("Cannot extract DyND C++ callable from None.")
    return &(c.v)

# returns a Python object, so no exception specifier is needed.
cdef callable wrap(const _callable &c):
    cdef callable cl = callable.__new__(callable)
    cl.v = c
    return cl

from ..ndt.type cimport wrap

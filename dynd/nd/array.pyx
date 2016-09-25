# cython: c_string_type=str, c_string_encoding=ascii

_builtin_type = type

from cpython.object cimport (Py_LT, Py_LE, Py_EQ, Py_NE, Py_GE, Py_GT,
                             PyObject_TypeCheck, PyTypeObject)
from cpython.buffer cimport PyObject_CheckBuffer
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp cimport bool as cpp_bool
from libcpp.complex cimport complex as cpp_complex
from cython.operator import dereference
from libcpp.vector cimport vector
import numpy as _np

from ..cpp.array cimport (groupby as dynd_groupby, empty as cpp_empty,
                          dtyped_zeros, dtyped_ones, dtyped_empty, array_and)
from ..cpp.arithmetic cimport pow
from ..cpp.type cimport make_type
from ..cpp.registry cimport registered
# from ..cpp.types.categorical_type cimport dynd_make_categorical_type
from ..cpp.types.datashape_formatter cimport format_datashape as dynd_format_datashape
from ..cpp.types.type_id cimport *
from ..cpp.view cimport view as _view
from ..pyobject_type cimport pyobject_id

from ..config cimport translate_exception
from ..ndt import type as ndt_type
from ..ndt.type cimport (type as _py_type, dynd_ndt_type_to_cpp, as_cpp_type,
                         cpp_type_for, _register_nd_array_type_deduction)

cdef extern from 'array_functions.hpp' namespace 'pydynd':
    void array_init_from_pyobject(_array&, object, object, bint, object) except +translate_exception
    void array_init_from_pyobject(_array&, object, object) except +translate_exception

    _array pyobject_array() except +translate_exception
    _array pyobject_array(object) except +translate_exception

    object array_get_shape(_array&) except +translate_exception
    object array_get_strides(_array&) except +translate_exception
    _array array_getitem(_array&, object) except +translate_exception
    void array_setitem(_array&, object, object) except +translate_exception
    object array_nonzero(_array&) except +translate_exception

    _array array_old_range(object, object, object, object) except +translate_exception
    _array array_old_linspace(object, object, object, object) except +translate_exception

    bint array_is_c_contiguous(_array&) except +translate_exception
    bint array_is_f_contiguous(_array&) except +translate_exception

    object array_as_numpy(object, bint) except +translate_exception

    const char *array_access_flags_string(_array&) except +translate_exception

    string array_repr(_array&) except +translate_exception

    _array dynd_parse_json_type(_type&, _array&, object) except +translate_exception
    void dynd_parse_json_array(_array&, _array&, object) except +translate_exception

    _array nd_fields(_array&, object) except +translate_exception

    int array_getbuffer_pep3118(object ndo, Py_buffer *buffer, int flags) except -1
    int array_releasebuffer_pep3118(object ndo, Py_buffer *buffer) except -1

cdef extern from "array_from_py.hpp" namespace "pydynd":
    void init_array_from_py() except *

cdef extern from 'numpy_interop.hpp' namespace 'pydynd':
    # Have Cython use an integer to represent the bool argument.
    # It will convert implicitly to bool at the C++ level.
    _array array_from_numpy_array_cast(PyObject*, unsigned int, bint)

cdef extern from 'init.hpp' namespace 'pydynd':
    void numpy_interop_init() except *

# Work around Cython misparsing various types when
# they are used as template parameters.
ctypedef long long longlong
ctypedef cpp_complex[double] cpp_complex_double

# Alias the builtin name `type` so it can be used in functions where it isn't
# in scope due to argument naming.
_builtin_type = type

# Initialize C level static interop data
numpy_interop_init()
init_array_from_py()

cdef class array(object):
    """
    nd.array(obj=None, dtype=None, type=None, access=None)

    Create a dynd array out of the provided object.

    The dynd array is the dynamically typed multi-dimensional
    object provided by the dynd library. It is similar to
    NumPy's _array, but has its dimensional structure encoded
    in the dynd type, along with the element type.

    When given a NumPy array, the resulting dynd array is a view
    into the NumPy array data. When given lists of Python object,
    an attempt is made to deduce an appropriate dynd type for
    the array, and a conversion is made if possible, or an
    exception is raised.

    Parameters
    ----------
    value : multi-dimensional object, optional
        Any object which dynd knows how to interpret as a dynd array.
    dtype: dynd type
        If provided, the type is used as the data type for the
        input, and the shape of the leading dimensions is deduced.
        This parameter cannot be used together with 'type'.
    type: dynd type
        If provided, the type is used as the full type for the input.
        If needed by the type, the shape is deduced from the input.
        This parameter cannot be used together with 'dtype'.
    access:  'readwrite'/'rw', 'readonly'/'r', or 'immutable', optional
        If provided, this specifies the access control for the
        created array. If the array is being allocated, as in
        construction from Python objects, this is the access control
        set.

        If the array is a view into another array data source,
        such as NumPy arrays or objects which support the buffer
        protocol, the access control must be compatible with that
        of the input object's.

        The default is immutable, or to inherit the access control
        of the object being viewed if it is an object supporting
        the buffer protocol.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> nd.array([1, 2, 3, 4, 5])
    nd.array([1, 2, 3, 4, 5],
             type="5 * int32")
    >>> nd.array([[1, 2], [3, 4, 5.0]])
    nd.array([   [1, 2], [3, 4, 5]],
             type="2 * var * float64")
    >>> from datetime import date
    >>> nd.array([date(2000,2,14), date(2012,1,1)])
    nd.array([2000-02-14, 2012-01-01],
             type="2 * date")
    """

    def __init__(self, value = None, type = None):

        if value is None and type is None:
            return

        cdef _type dst_tp
        if type is None:
            dst_tp = cpp_type_for(value)
            self.v = cpp_empty(dst_tp)
            self.v.assign(pyobject_array(value))
        else:
            if (not isinstance(type, ndt_type)):
                type = ndt_type(type)
            dst_tp = dynd_ndt_type_to_cpp(type)
            self.v = cpp_empty(dst_tp)
            self.v.assign(pyobject_array(value))

    property access_flags:
        """
        a.access_flags
        The access flags of the dynd array, as a string.
        Returns 'immutable', 'readonly', or 'readwrite'
        """
        def __get__(self):
            return str(<char *>array_access_flags_string(self.v))

    property ndim:
        """
        tp.ndim
        The number of array dimensions in this dynd type.
        This property is like NumPy
        _array's 'ndim'. Indexing with [] can in many cases
        go deeper than just the array dimensions, for
        example structs can be indexed this way.
        """
        def __get__(self):
            return self.v.get_ndim()

    property strides:
        def __get__(self):
            return array_get_strides(self.v)

    property shape:
        def __get__(self):
            return array_get_shape(self.v)

    property dtype:
        def __get__(self):
            cdef _py_type result = _py_type()
            result.v = self.v.get_dtype()
            return result

    def to(self, tp = None):
        """
        nd.to(tp)

        Evaluates the dynd array, and converts it into a NumPy object.

        Parameters
        ----------
        n : dynd array
            The array to convert into native Python types.
        allow_copy : bool, optional
            If true, allows a copy to be made when the array types
            can't be directly viewed as a NumPy array, but with a
            data-preserving copy they can be.

        Examples
        --------
        >>> from dynd import nd, ndt
        >>> import numpy as np
        >>> a = nd.array([[1, 2, 3], [4, 5, 6]])
        >>> a
        nd.array([[1, 2, 3], [4, 5, 6]],
                type="2 * 3 * int32")
        >>> nd.as_numpy(a)
        array([[1, 2, 3],
            [4, 5, 6]])
        """

        try:
            import numpy as np

            if (tp == np.ndarray):
                return array_as_numpy(self, bool(True))
        except Exception:
            pass

        raise ValueError('could not copy to type ' + tp)

    def __contains__(self, x):
        raise NotImplementedError('__contains__ is not yet implemented for nd.array')

    def __dir__(self):
        # Customize dir() so that additional properties of various types
        # will show up in IPython tab-complete, for example.
        result = dict(array.__dict__)
        result.update(object.__dict__)

        for pair in registered():
            result[pair.first] = wrap(pair.second.value())

        return result.keys()

    def __getattr__(self, name):
        if self.v.is_null():
            raise AttributeError(name)

        for pair in registered('dynd.nd'):
            if (pair.first == <string> name):
                return dynd_nd_array_from_cpp(pair.second.value()(self.v))

        try:
            return dynd_nd_array_from_cpp(self.v.p(name))
        except ValueError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if self.v.is_null():
            raise AttributeError(name)

        self.v.p(name).assign(pyobject_array(value))

    def __getitem__(self, x):
        from .. import ndt, nd

        cdef array idx
        if isinstance(x, list):
          idx = asarray(x)
          if (idx.dtype == ndt.bool):
            return nd.take(self, idx)
        elif isinstance(x, array):
          if (x.dtype == ndt.bool):
            return nd.take(self, x)

        cdef array result = array()
        result.v = array_getitem(self.v, x)
        return result

    def __index__(array self):
        cdef type_id_t tp = \
            dynd_nd_array_to_cpp(self).get_type().get_base_id()
        if tp == uint_kind_id or tp == int_kind_id:
            return dynd_nd_array_to_cpp(self).as[longlong]()
        raise TypeError('Only integer scalars can be converted '
                        'to scalar indices.')

    def __int__(array self):
        cdef type_id_t tp = \
            dynd_nd_array_to_cpp(self).get_type().get_base_id()
        if (tp == uint_kind_id or tp == int_kind_id or
            tp == bool_kind_id):
            return dynd_nd_array_to_cpp(self).as[longlong]()
        raise TypeError('Only integer and boolean scalars can be '
                        'converted to integers.')

    def __float__(array self):
        cdef type_id_t tp = \
            dynd_nd_array_to_cpp(self).get_type().get_base_id()
        if (tp == uint_kind_id or tp == int_kind_id or
            tp == bool_kind_id or tp == float_kind_id):
            return dynd_nd_array_to_cpp(self).as[double]()
        raise TypeError('Only integer, boolean, and floating point '
                        'scalars can be converted to floating point numbers.')

    def __complex__(self):
        cdef type_id_t tp = \
            dynd_nd_array_to_cpp(self).get_type().get_base_id()
        cdef cpp_complex_double ret
        if (tp == uint_kind_id or tp == int_kind_id or
            tp == bool_kind_id or tp == float_kind_id
            or tp == complex_kind_id):
            ret = dynd_nd_array_to_cpp(self).as[cpp_complex_double]()
            return complex(ret.real(), ret.imag())
        raise TypeError('Only integer, boolean, floating point, and '
                        'complex floating point scalars can be converted '
                        'to complex floating point numbers.')

    def __len__(self):
        return self.v.get_dim_size()

    def __nonzero__(self):
        return array_nonzero(self.v)

    def __repr__(self):
        return str(<char *>array_repr(self.v).c_str())

    def __setitem__(self, x, y):
        array_setitem(self.v, x, y)

    def __pos__(self):
        return dynd_nd_array_from_cpp(+as_cpp_array(self))

    def __neg__(self):
        return dynd_nd_array_from_cpp(-as_cpp_array(self))

    def __invert__(self):
        return dynd_nd_array_from_cpp(~as_cpp_array(self))

    def __add__(lhs, rhs):
        return dynd_nd_array_from_cpp(
            as_cpp_array(lhs) + as_cpp_array(rhs))

    def __radd__(rhs, lhs):
        return dynd_nd_array_from_cpp(
            as_cpp_array(lhs) + as_cpp_array(rhs))

    def __sub__(lhs, rhs):
        return dynd_nd_array_from_cpp(
            as_cpp_array(lhs) - as_cpp_array(rhs))

    def __rsub__(rhs, lhs):
        return dynd_nd_array_from_cpp(
            as_cpp_array(lhs) - as_cpp_array(rhs))

    def __mul__(lhs, rhs):
        return dynd_nd_array_from_cpp(
            as_cpp_array(lhs) * as_cpp_array(rhs))

    def __rmul__(rhs, lhs):
        return dynd_nd_array_from_cpp(
            as_cpp_array(lhs) * as_cpp_array(rhs))

    def __div__(lhs, rhs):
        return dynd_nd_array_from_cpp(
            as_cpp_array(lhs) / as_cpp_array(rhs))

    def __rdiv__(rhs, lhs):
        return dynd_nd_array_from_cpp(
            as_cpp_array(lhs) / as_cpp_array(rhs))

    def __truediv__(lhs, rhs):
        return dynd_nd_array_from_cpp(
            as_cpp_array(lhs) / as_cpp_array(rhs))

    def __rtruediv__(rhs, lhs):
        return dynd_nd_array_from_cpp(
            as_cpp_array(lhs) / as_cpp_array(rhs))

    def __mod__(lhs, rhs):
        return dynd_nd_array_from_cpp(
            as_cpp_array(lhs) % as_cpp_array(rhs))

    def __rmod__(rhs, lhs):
        return dynd_nd_array_from_cpp(
            as_cpp_array(lhs) % as_cpp_array(rhs))

    def __and__(lhs, rhs):
        return dynd_nd_array_from_cpp(
            array_and(as_cpp_array(lhs), as_cpp_array(rhs)))

    def __rand__(rhs, lhs):
        return dynd_nd_array_from_cpp(
            array_and(as_cpp_array(lhs), as_cpp_array(rhs)))

    def __or__(lhs, rhs):
        return dynd_nd_array_from_cpp(
            as_cpp_array(lhs) | as_cpp_array(rhs))

    def __ror__(rhs, lhs):
        return dynd_nd_array_from_cpp(
            as_cpp_array(lhs) | as_cpp_array(rhs))

    def __xor__(lhs, rhs):
        return dynd_nd_array_from_cpp(
            as_cpp_array(lhs) ^ as_cpp_array(rhs))

    def __rxor__(rhs, lhs):
        return dynd_nd_array_from_cpp(
            as_cpp_array(lhs) ^ as_cpp_array(rhs))

    def __lshift__(lhs, rhs):
        return dynd_nd_array_from_cpp(
            as_cpp_array(lhs) << as_cpp_array(rhs))

    def __rlshift__(rhs, lhs):
        return dynd_nd_array_from_cpp(
            as_cpp_array(lhs) << as_cpp_array(rhs))

    def __rshift__(lhs, rhs):
        return dynd_nd_array_from_cpp(
            as_cpp_array(lhs) >> as_cpp_array(rhs))

    def __rrshift__(rhs, lhs):
        return dynd_nd_array_from_cpp(
            as_cpp_array(lhs) >> as_cpp_array(rhs))

    def __pow__(lhs, rhs, mod_base):
        if mod_base is not None:
            raise ValueError("Support for exponentiation modulo a "
                             "given value is not currently implemented.")
        return dynd_nd_array_from_cpp(pow(
            as_cpp_array(lhs), as_cpp_array(rhs)))

    def __rpow__(rhs, lhs):
        return dynd_nd_array_from_cpp(pow(
            as_cpp_array(lhs), as_cpp_array(rhs)))

    def __richcmp__(a0, a1, int op):
        if op == Py_LT:
            return dynd_nd_array_from_cpp(asarray(a0).v < asarray(a1).v)
        elif op == Py_LE:
            return dynd_nd_array_from_cpp(asarray(a0).v <= asarray(a1).v)
        elif op == Py_EQ:
            return dynd_nd_array_from_cpp(asarray(a0).v == asarray(a1).v)
        elif op == Py_NE:
            return dynd_nd_array_from_cpp(asarray(a0).v != asarray(a1).v)
        elif op == Py_GE:
            return dynd_nd_array_from_cpp(asarray(a0).v >= asarray(a1).v)
        elif op == Py_GT:
            return dynd_nd_array_from_cpp(asarray(a0).v > asarray(a1).v)

    def __getbuffer__(array self, Py_buffer* buffer, int flags):
        # Docstring triggered Cython bug (fixed in master), so it's commented out
        #"""PEP 3118 buffer protocol"""
        array_getbuffer_pep3118(self, buffer, flags)

    def __releasebuffer__(array self, Py_buffer* buffer):
        # Docstring triggered Cython bug (fixed in master), so it's commented out
        #"""PEP 3118 buffer protocol"""
        array_releasebuffer_pep3118(self, buffer)

    def cast(array self, tp):
        """
        a.cast(type)
        Casts the dynd array's type to the requested type,
        producing a conversion type. If the data for the
        new type is identical, it is used directly to avoid
        the conversion.
        Parameters
        ----------
        type : dynd type
            The type is cast into this type.
        """
        cdef _type t = as_cpp_type(tp)
        cdef _array res = cpp_empty(t)
        res.assign(dynd_nd_array_to_cpp(self))

        return dynd_nd_array_from_cpp(res)

    def eval(array self):
        """
        a.eval()
        Returns a version of the dynd array with plain values,
        all expressions evaluated. This returns the original
        array back if it has no expression type.
        Examples
        --------
        >>> from dynd import nd, ndt
        >>> a = nd.array([1.5, 2, 3])
        >>> a
        nd.array([1.5,   2,   3],
                 type="3 * float64")
        >>> b = a.ucast(ndt.complex_float32)
        >>> b
        nd.array([(1.5 + 0j),   (2 + 0j),   (3 + 0j)],
                 type="3 * convert[to=complex[float32], from=float64]")
        >>> b.eval()
        nd.array([(1.5 + 0j),   (2 + 0j),   (3 + 0j)],
                 type="3 * complex[float32]")
        """
        return dynd_nd_array_from_cpp(dynd_nd_array_to_cpp(self).eval())

    def sum(self, axis = None):
      from .. import nd

      if (axis is None):
        return nd.sum(self)

      return nd.sum(self, axes = [axis])

    def ucast(array self, dtype, ssize_t replace_ndim=0):
        """
        a.ucast(dtype, replace_ndim=0)
        Casts the dynd array's dtype to the requested type,
        producing a conversion type. The dtype is the type
        after the nd.ndim_of(a) array dimensions.
        Parameters
        ----------
        dtype : dynd type
            The dtype of the array is cast into this type.
            If `replace_ndim` is not zero, then that many
            dimensions are included in what is cast as well.
        replace_ndim : integer, optional
            The number of array dimensions to replace in doing
            the cast.
        Examples
        --------
        >>> from dynd import nd, ndt
        >>> from datetime import date
        >>> a = nd.array([date(1929,3,13), date(1979,3,22)]).ucast('{month: int32, year: int32, day: float32}')
        >>> a
        nd.array([[3, 1929, 13], [3, 1979, 22]], type="Fixed * convert[to={month : int32, year : int32, day : float32}, from=date]")
        >>> a.eval()
        nd.array([[3, 1929, 13], [3, 1979, 22]], type="2 * {month : int32, year : int32, day : float32}")
        """
        cdef _type t = as_cpp_type(dtype)
        return dynd_nd_array_from_cpp(dynd_nd_array_to_cpp(self).ucast(t, replace_ndim))

    def view_scalars(self, dtp):
        """
        a.view_scalars(dtype)
        Views the data of the dynd array as the requested dtype,
        where it makes sense.
        If the array is a one-dimensional contiguous
        array of plain old data, the new dtype may have a different
        element size than original one.
        When the array has an expression type, the
        view is created by layering another view dtype
        onto the array's existing expression.
        Parameters
        ----------
        dtype : dynd type
            The scalars are viewed as this dtype.
        """
        cdef array result = array()
        result.v = self.v.view_scalars(_py_type(dtp).v)
        return result

cdef _array dynd_nd_array_to_cpp(array a) nogil except *:
    # Once this becomes a method of the type wrapper class, this check and
    # its corresponding exception handler declaration are no longer necessary
    # since the self parameter is guaranteed to never be None.
    if a is not None:
        return a.v
    with gil:
        raise TypeError("Cannot extract DyND C++ array from None.")

cdef _array dynd_nd_array_to_cpp_unsafe(array a) nogil:
    return a.v

cdef _array *dynd_nd_array_to_ptr(array a) nogil except *:
    # Once this becomes a method of the type wrapper class, this check and
    # its corresponding exception handler declaration are no longer necessary
    # since the self parameter is guaranteed to never be None.
    if a is not None:
        return &(a.v)
    with gil:
        raise TypeError("Cannot extract DyND C++ array from None.")

# returns a Python object, so no exception specifier is needed.
cdef array dynd_nd_array_from_cpp(_array a):
    cdef array arr = array.__new__(array)
    arr.v = a
    return arr

cdef _type _type_from_pyarr_wrapper(PyObject *a) nogil:
    return dynd_nd_array_to_cpp_unsafe(<array>a).get_type()

_register_nd_array_type_deduction(<PyTypeObject*>array, &_type_from_pyarr_wrapper)

cdef _array as_cpp_array(object obj) except *:
    """
    nd.as_cpp_array(obj)
    Constructs a dynd array from a Python object, taking
    a view if possible, otherwise making a copy. If the object
    is already a DyND array, this is equivalent to calling
    dynd_nd_array_to_cpp(obj).
    """
    # TODO: Remove lazy import since it's really only needed because of the weird
    # boxing and unboxing used further down
    from . import assign
    if _builtin_type(obj) is array:
        return dynd_nd_array_to_cpp(obj)
    elif _builtin_type(obj) is _np.ndarray:
        return array_from_numpy_array_cast(<PyObject*>obj, 0, 0)
    # elif PyObject_CheckBuffer(obj):
    #     TODO
    cdef _type tp = cpp_type_for(obj)
    cdef _array out = cpp_empty(tp)
    out.assign(pyobject_array(obj))
    return out

cpdef array asarray(object obj):
    """
    nd.asarray(obj)
    Constructs a dynd array from the object, taking a view
    if possible, otherwise making a copy. If the object is already
    a DyND array, the same object is returned.
    Parameters
    ----------
    obj : object
        The object which is to be converted into a dynd array,
        as a view if possible, otherwise a copy.
    """
    if _builtin_type(obj) is array:
        return obj
    return dynd_nd_array_from_cpp(as_cpp_array(obj))

from dynd.nd.callable cimport callable
from cython.operator cimport dereference

def type_of(a):
    """
    nd.type_of(a)
    The dynd type of the array. This is the full
    data type, including its multi-dimensional structure.
    Parameters
    ----------
    a : dynd array
        The array whose type is requested.
    Examples
    --------
    >>> from dynd import nd, ndt
    >>> nd.type_of(nd.array([1,2,3,4]))
    ndt.type("4 * int32")
    >>> nd.type_of(nd.array([[1,2],[3.0]]))
    ndt.type("2 * var * float64")
    """
    cdef _py_type result = _py_type()
    if isinstance(a, array):
        result.v = (<array> a).v.get_type()
    elif isinstance(a, callable):
        result.v = dereference((<callable> a).v).get_type()

    return result

def dtype_of(array a, size_t include_ndim=0):
    """
    nd.dtype_of(a)
    The data type of the dynd array. This is
    the type after removing all the array
    dimensions from the dynd type of `a`.
    If `include_ndim` is provided, that many
    array dimensions are included in the
    data type returned.
    Parameters
    ----------
    a : dynd array
        The array whose type is requested.
    include_ndim : int, optional
        The number of array dimensions to include
        in the data type, default zero.
    """
    cdef _py_type result = _py_type()
    result.v = a.v.get_dtype(include_ndim)
    return result

def dshape_of(array a):
    """
    nd.dshape_of(a)
    The blaze datashape of the dynd array, as a string.
    Parameters
    ----------
    a : dynd array
        The array whose type is requested.
    """
    return str(<char *>dynd_format_datashape(a.v.get_type()).c_str())

def ndim_of(array a):
    """
    nd.ndim_of(a)
    The number of array dimensions in the dynd array `a`.
    This corresponds to the number of dimensions
    in a NumPy array.
    """
    return a.v.get_ndim()

def is_c_contiguous(array a):
    """
    nd.is_c_contiguous(a)
    Returns True if the array is C-contiguous, False
    otherwise. An array is C-contiguous if all its array
    dimensions are ``fixed``, the strides are in decreasing
    order, and the data is tightly packed.
    """
    return array_is_c_contiguous(a.v)

def is_f_contiguous(array a):
    """
    nd.is_f_contiguous(a)
    Returns True if the array is F-contiguous, False
    otherwise. An array is F-contiguous if all its array
    dimensions are ``fixed``, the strides are in increasing
    order, and the data is tightly packed.
    """
    return array_is_f_contiguous(a.v)

def as_py(array n):
    """
    nd.as_py(n)
    Evaluates the dynd array, converting it into native Python types.
    Uniform dimensions convert into Python lists, struct types convert
    into Python dicts, scalars convert into the most appropriate Python
    scalar for their type.
    Parameters
    ----------
    n : dynd array
        The dynd array to convert into native Python types.
    tuple : bool
        If true, produce tuples instead of dicts when converting
        dynd struct arrays.
    Examples
    --------
    >>> from dynd import nd, ndt
    >>> a = nd.array([1, 2, 3, 4.0])
    >>> a
    nd.array([1, 2, 3, 4],
             type="4 * float64")
    >>> nd.as_py(a)
    [1.0, 2.0, 3.0, 4.0]
    """
    cdef _array res = pyobject_array(None)
    res.assign(dynd_nd_array_to_cpp(n))
    return <object> dereference(<PyObject **> res.data())

def view(obj, type=None):
    """
    nd.view(obj, type=None)
    Constructs a dynd array which is a view of the data from
    `obj`. If a type for the returned array is not provided,
    the type of `obj` is used.
    Parameters
    ----------
    obj : object
        A Python object which backs some array data, such as
        a dynd array, a numpy array, or an object supporting
        the Python buffer protocol.
    type : ndt.type, optional
        If provided, requests that the memory of ``obj`` be viewed
        as this type.
    """
    if not PyObject_CheckBuffer(obj):
        raise TypeError('Python objects that do not support the buffer '
                        'protocol cannot be viewed as DyND arrays.')
    cdef _array input = dynd_nd_array_to_cpp(asarray(obj))
    # TODO: determine if there is a good way to have the branching be done
    # on the C++ side of things for the case where the type is not provided.
    if type is None:
        return dynd_nd_array_from_cpp(input)
    cdef _type tp = dynd_ndt_type_to_cpp(_py_type(type))
    return dynd_nd_array_from_cpp(_view(input, tp))

# TODO: The wrappers for zeros, ones, and empty are identical.
# They should be handled via a macro of some sort (Use Tempita?)
# Some expansion of the interface on the C++ side is probably necessary
# to allow that to work easily. That needs to happen anyway.

def zeros(*args):
    """
    nd.zeros(type)
    nd.zeros(shape, type)
    nd.zeros(shape_0, shape_1, ..., shape_(n-1), type)
    Creates an array of zeros of the specified
    type. Dimensions may be provided as integer
    positional parameters, a tuple of integers,
    or within the type itself.
    TODO: In the immutable case it should use zero-strides to optimize storage.
    TODO: Add order= keyword-only argument. This would accept
    'C', 'F', or a permutation tuple.
    Parameters
    ----------
    shape : list of int, optional
        If provided, specifies the shape for dimensions which
        are prepended to the following dtype.
    type : dynd type
        The type of the uninitialized array to create.
    """
    cdef size_t largs = len(args)
    cdef _array ret
    if largs  == 1:
        # Only the full type is provided
        tp = args[0]
        if _builtin_type(tp) in [int, long]:
            raise ValueError('Data type must be explicitly specified. '
                             'It cannot be provided as an integer.')
        ret = cpp_empty(as_cpp_type(tp))
        ret.assign(_array(0))
        return dynd_nd_array_from_cpp(ret)
    # TODO: This should be a size_t, not a ssize_t, but the C++ interface needs
    # to be updtated to support that.
    cdef vector[ssize_t] shape
    if largs == 2:
        # The shape is a provided as a tuple (or single integer)
        tp = args[1]
        if _builtin_type(tp) in [int, long]:
            raise ValueError('Data type must be explicitly specified. '
                             'It cannot be provided as an integer.')
        py_shape = args[0]
        if _builtin_type(py_shape) in [int, long]:
            py_shape = (py_shape,)
        shape = py_shape
        # TODO: C++ interface should be using size_t instead of ssize_t
        # here as well.
        ret = dtyped_zeros(<ssize_t>shape.size(), shape.data(), as_cpp_type(tp))
        return dynd_nd_array_from_cpp(ret)
    if largs > 2:
        # The shape is expanded out in the arguments
        tp = args[-1]
        shape = args[:-1]
        if _builtin_type(tp) in [int, long]:
            raise ValueError('Data type must be explicitly specified. '
                             'It cannot be provided as an integer.')
        ret = dtyped_zeros(shape.size(), shape.data(), as_cpp_type(tp))
        return dynd_nd_array_from_cpp(ret)
    raise TypeError('nd.zeros() expected at least 1 positional argument, got 0')

def ones(*args, **kwargs):
    """
    nd.ones(type, *)
    nd.ones(shape, dtype, *)
    nd.ones(shape_0, shape_1, ..., shape_(n-1), dtype, *)
    Creates an array of ones of the specified
    type. Dimensions may be provided as integer
    positional parameters, a tuple of integers,
    or within the dtype itself.
    TODO: In the immutable case it should use zero-strides to optimize storage.
    TODO: Add order= keyword-only argument. This would accept
    'C', 'F', or a permutation tuple.
    Parameters
    ----------
    shape : list of int, optional
        If provided, specifies the shape for dimensions which
        are prepended to the following dtype.
    dtype : dynd type
        The dtype of the uninitialized array to create.
    """
    cdef size_t largs = len(args)
    cdef _array ret
    if largs  == 1:
        # Only the full type is provided
        tp = args[0]
        if _builtin_type(tp) in [int, long]:
            raise ValueError('Data type must be explicitly specified. '
                             'It cannot be provided as an integer.')
        ret = cpp_empty(as_cpp_type(tp))
        ret.assign(_array(1))
        return dynd_nd_array_from_cpp(ret)
    # TODO: This should be a size_t, not a ssize_t, but the C++ interface needs
    # to be updtated to support that.
    cdef vector[ssize_t] shape
    if largs == 2:
        # The shape is a provided as a tuple (or single integer)
        tp = args[1]
        if _builtin_type(tp) in [int, long]:
            raise ValueError('Data type must be explicitly specified. '
                             'It cannot be provided as an integer.')
        py_shape = args[0]
        if _builtin_type(py_shape) in [int, long]:
            py_shape = (py_shape,)
        shape = py_shape
        # TODO: C++ interface should be using size_t instead of ssize_t
        # here as well.
        ret = dtyped_ones(<ssize_t>shape.size(), shape.data(), as_cpp_type(tp))
        return dynd_nd_array_from_cpp(ret)
    if largs > 2:
        # The shape is expanded out in the arguments
        tp = args[-1]
        shape = args[:-1]
        if _builtin_type(tp) in [int, long]:
            raise ValueError('Data type must be explicitly specified. '
                             'It cannot be provided as an integer.')
        ret = dtyped_ones(shape.size(), shape.data(), as_cpp_type(tp))
        return dynd_nd_array_from_cpp(ret)
    raise TypeError('nd.ones() expected at least 1 positional argument, got 0')

def empty(*args, **kwargs):
    """
    nd.empty(type)
    nd.empty(shape, dtype)
    nd.empty(shape_0, shape_1, ..., shape_(n-1), dtype)
    Creates an uninitialized array of the specified type.
    Dimensions may be provided as integer positional
    parameters, a tuple of integers, or within the dtype itself.
    TODO: Add order= keyword-only argument. This would accept
    'C', 'F', or a permutation tuple.
    Parameters
    ----------
    shape : list of int, optional
        If provided, specifies the shape for dimensions which
        are prepended to the following dtype.
    dtype : dynd type
        The dtype of the uninitialized array to create.
    Examples
    --------
    >>> from dynd import nd, ndt
    >>> nd.empty('2 * 2 * int8')
    nd.array([[0, -14], [0, 0]],
             type="2 * 2 * int8")
    >>> nd.empty((2, 2), ndt.int16)
    nd.array([[0, 27], [159, 0]],
             type="2 * 2 * int16")
    """
    cdef size_t largs = len(args)
    cdef _array ret
    if largs  == 1:
        # Only the full type is provided
        tp = args[0]
        if _builtin_type(tp) in [int, long]:
            raise ValueError('Data type must be explicitly specified. '
                             'It cannot be provided as an integer.')
        ret = cpp_empty(as_cpp_type(tp))
        return dynd_nd_array_from_cpp(ret)
    # TODO: This should be a size_t, not a ssize_t, but the C++ interface needs
    # to be updtated to support that.
    cdef vector[ssize_t] shape
    if largs == 2:
        # The shape is a provided as a tuple (or single integer)
        tp = args[1]
        if _builtin_type(tp) in [int, long]:
            raise ValueError('Data type must be explicitly specified. '
                             'It cannot be provided as an integer.')
        py_shape = args[0]
        if _builtin_type(py_shape) in [int, long]:
            py_shape = (py_shape,)
        shape = py_shape
        # TODO: C++ interface should be using size_t instead of ssize_t
        # here as well.
        ret = dtyped_empty(<ssize_t>shape.size(), shape.data(), as_cpp_type(tp))
        return dynd_nd_array_from_cpp(ret)
    if largs > 2:
        # The shape is expanded out in the arguments
        tp = args[-1]
        shape = args[:-1]
        if _builtin_type(tp) in [int, long]:
            raise ValueError('Data type must be explicitly specified. '
                             'It cannot be provided as an integer.')
        ret = dtyped_empty(shape.size(), shape.data(), as_cpp_type(tp))
        return dynd_nd_array_from_cpp(ret)
    raise TypeError('nd.empty() expected at least 1 positional argument, got 0')

def old_range(start=None, stop=None, step=None, dtype=None):
    """
    nd.old_range(stop, dtype=None)
    nd.old_range(start, stop, step=None, dtype=None)
    Constructs a dynd array representing a stepped range of values.
    This function assumes that (stop-start)/step is approximately
    an integer, so as to be able to produce reasonable values with
    fractional steps which can't be exactly represented, such as 0.1.
    Parameters
    ----------
    start : int, optional
        If provided, this is the first value in the resulting dynd array.
    stop : int
        This provides the stopping criteria for the range, and is
        not included in the resulting dynd array.
    step : int
        This is the increment.
    dtype : dynd type, optional
        If provided, it must be a scalar type, and the result
        is of this type.
    """
    cdef array result = array()
    # Move the first argument to 'stop' if stop isn't specified
    if stop is None:
        if start is not None:
            result.v = array_old_range(None, start, step, dtype)
        else:
            raise ValueError("No value provided for 'stop'")
    else:
        result.v = array_old_range(start, stop, step, dtype)
    return result

def old_linspace(start, stop, count=50, dtype=None):
    """
    nd.old_linspace(start, stop, count=50, dtype=None)
    Constructs a specified count of values interpolating a range.
    Parameters
    ----------
    start : floating point scalar
        The value of the first element of the resulting dynd array.
    stop : floating point scalar
        The value of the last element of the resulting dynd array.
    count : int, optional
        The number of elements in the resulting dynd array.
    dtype : dynd type, optional
        If provided, it must be a scalar type, and the result
        is of this type.
    """
    cdef array result = array()
    result.v = array_old_linspace(start, stop, count, dtype)
    return result

def parse_json(tp, json, ectx=None):
    """
    nd.parse_json(type, json, ectx)
    Parses an input JSON string as a particular dynd type.
    Parameters
    ----------
    type : dynd type
        The type to interpret the input JSON as. If the data
        does not match this type, an error is raised during parsing.
    json : string or bytes
        String that contains the JSON to parse.
    ectx : eval_context, optional
        If provided an evaluation context to use when processing the JSON.
    Examples
    --------
    >>> from dynd import nd, ndt
    >>> nd.parse_json('var * int8', '[1, 2, 3, 4, 5]')
    nd.array([1, 2, 3, 4, 5],
             type="var * int8")
    >>> nd.parse_json('4 * int8', '[1, 2, 3, 4]')
    nd.array([1, 2, 3, 4],
             type="4 * int8")
    >>> nd.parse_json('2 * {x: int8, y: int8}', '[{"x":0, "y":1}, {"y":2, "x":3}]')
    nd.array([[0, 1], [3, 2]],
             type="2 * {x : int8, y : int8}")
    """
    cdef array result = array()
    if isinstance(tp, array):
        dynd_parse_json_array((<array>tp).v, array(json).v, ectx)
    else:
        result.v = dynd_parse_json_type(_py_type(tp).v, array(json).v, ectx)
        return result

import operator

def _validate_squeeze_index(i, sz):
    try:
        i = operator.index(i)
    except TypeError:
        raise TypeError('nd.squeeze() requires an int or '
                    + 'tuple of ints for axis parameter')
    if i >= 0:
        if i >= sz:
            raise IndexError('nd.squeeze() axis %d is out of range' % i)
    else:
        if i < -sz:
            raise IndexError('nd.squeeze() axis %d is out of range' % i)
        else:
            i += sz
    return i

def squeeze(a, axis=None):
    """Removes size-one dimensions from the beginning and end of the shape.
    If `axis` is provided, removes the dimensions specified in the tuple.
    Parameters
    ----------
    a : nd.array
        The array to be squeezed.
    axis : int or tuple of int, optional
        Specifies which exact dimensions to remove. The dimensions specified
        must have size one.
    """
    s = a.shape
    ssz = len(s)
    ix = [slice(None)]*ssz
    if axis is not None:
        if isinstance(axis, tuple):
            axis = [_validate_squeeze_index(x, ssz) for x in axis]
        else:
            axis = [_validate_squeeze_index(axis, ssz)]
        for x in axis:
            if s[x] != 1:
                raise IndexError(('nd.squeeze() requested axis %d ' +
                        'has shape %d, not 1 as required') % (x, s[x]))
            ix[x] = 0
    else:
        # Construct a list of indexer objects which trim off the
        # beginning and end
        for i in range(ssz):
            if s[i] == 1:
                ix[i] = 0
            else:
                break
        for i in range(ssz-1, -1, -1):
            if s[i] == 1:
                ix[i] = 0
            else:
                break
    ix = tuple(ix)
    return a[ix]

def fields(array struct_array, *fields_list):
    """
    nd.fields(struct_array, *fields_list)
    Selects fields from an array of structs.
    Parameters
    ----------
    struct_array : dynd array with struct dtype
        A dynd array whose dtype has kind 'struct'. This
        could be a single struct instance, or an array of structs.
    *fields_list : string
        The remaining parameters must all be strings, and are the field
        names to select.
    """
    cdef array result = array()
    result.v = nd_fields(struct_array.v, fields_list)
    return result

from .callable cimport wrap

# These are functions exported by the numpy interop stuff that are needed in
# other modules. These are only here until we put together a better way of
# making them available where they are actually needed.

cdef extern from 'functional.hpp':
    _callable _apply 'apply'(_type, object) except +translate_exception

cdef _callable _functional_apply(_type t, object o) except *:
    return _apply(t, o)

cdef extern from 'assign.hpp':
    void assign_init() except +translate_exception

cdef void _registry_assign_init() except *:
    assign_init()

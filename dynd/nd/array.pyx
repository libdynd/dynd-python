# cython: c_string_type=str, c_string_encoding=ascii

from cpython.object cimport Py_LT, Py_LE, Py_EQ, Py_NE, Py_GE, Py_GT
from libcpp.string cimport string

from ..cpp.array cimport groupby as dynd_groupby
from ..cpp.type cimport type as _type
from ..cpp.types.categorical_type cimport dynd_make_categorical_type
from ..cpp.types.datashape_formatter cimport format_datashape as dynd_format_datashape

from ..config cimport translate_exception
from ..wrapper cimport set_wrapper_type, wrap
from ..ndt.type cimport type
from ..ndt import Unsupplied

cdef extern from 'array_functions.hpp' namespace 'pydynd':
    void init_w_array_typeobject(object)

    void array_init_from_pyobject(_array&, object, object, bint, object) except +translate_exception
    void array_init_from_pyobject(_array&, object, object) except +translate_exception

    object array_int(_array&) except +translate_exception
    object array_float(_array&) except +translate_exception
    object array_complex(_array&) except +translate_exception

    _array array_asarray(object, object) except +translate_exception
    object array_get_shape(_array&) except +translate_exception
    object array_get_strides(_array&) except +translate_exception
    _array array_getitem(_array&, object) except +translate_exception
    void array_setitem(_array&, object, object) except +translate_exception
    _array array_view(object, object, object) except +translate_exception
    _array array_zeros(_type&, object) except +translate_exception
    _array array_zeros(object, _type&, object) except +translate_exception
    _array array_ones(_type&, object) except +translate_exception
    _array array_ones(object, _type&, object) except +translate_exception
    _array array_full(_type&, object, object) except +translate_exception
    _array array_full(object, _type&, object, object) except +translate_exception
    _array array_empty(_type&, object) except +translate_exception
    _array array_empty(object, _type&, object) except +translate_exception
    object array_index(_array&) except +translate_exception
    object array_nonzero(_array&) except +translate_exception

    _array array_eval(_array&, object) except +translate_exception
    _array array_cast(_array&, _type&) except +translate_exception
    _array array_ucast(_array&, _type&, size_t) except +translate_exception
    _array array_range(object, object, object, object) except +translate_exception
    _array array_linspace(object, object, object, object) except +translate_exception

    bint array_is_c_contiguous(_array&) except +translate_exception
    bint array_is_f_contiguous(_array&) except +translate_exception

    object array_as_py(_array&, bint) except +translate_exception
    object array_as_numpy(object, bint) except +translate_exception

    const char *array_access_flags_string(_array&) except +translate_exception

    string array_repr(_array&) except +translate_exception
    object array_str(_array&) except +translate_exception
    object array_unicode(_array&) except +translate_exception

    _array array_add(_array&, _array&) except +translate_exception
    _array array_subtract(_array&, _array&) except +translate_exception
    _array array_multiply(_array&, _array&) except +translate_exception
    _array array_divide(_array&, _array&) except +translate_exception

    _array dynd_parse_json_type(_type&, _array&, object) except +translate_exception
    void dynd_parse_json_array(_array&, _array&, object) except +translate_exception

    _array nd_fields(_array&, object) except +translate_exception

    int array_getbuffer_pep3118(object ndo, Py_buffer *buffer, int flags) except -1
    int array_releasebuffer_pep3118(object ndo, Py_buffer *buffer) except -1

cdef extern from 'gfunc_callable_functions.hpp' namespace 'pydynd':
    object get_array_dynamic_property(_array&, object) except +translate_exception
    void add_array_names_to_dir_dict(_array&, object) except +translate_exception
    void set_array_dynamic_property(_array&, object, object) except +translate_exception

    cdef cppclass array_callable_wrapper:
        pass
    object array_callable_call(array_callable_wrapper&, object, object) except +translate_exception
    void init_w_array_callable_typeobject(object)

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

    def __init__(self, value=Unsupplied, dtype=None, type=None, access=None):
        if value is not Unsupplied:
            # Get the array data
            if dtype is not None:
                if type is not None:
                    raise ValueError('Must provide only one of ' +
                                    'dtype or type, not both')
                array_init_from_pyobject(self.v, value, dtype, False, access)
            elif type is not None:
                array_init_from_pyobject(self.v, value, type, True, access)
            else:
                array_init_from_pyobject(self.v, value, access)
        elif dtype is not None or type is not None or access is not None:
            raise ValueError('a value for the array construction must ' +
                            'be provided when another keyword parameter is used')

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
            cdef type result = type()
            result.v = self.v.get_dtype()
            return result

    def __contains__(self, x):
        raise NotImplementedError('__contains__ is not yet implemented for nd.array')

    def __dir__(self):
        # Customize dir() so that additional properties of various types
        # will show up in IPython tab-complete, for example.
        result = dict(array.__dict__)
        result.update(object.__dict__)
        add_array_names_to_dir_dict(self.v, result)
        return result.keys()

    def __getattr__(self, name):
        return get_array_dynamic_property(self.v, name)

    def __setattr__(self, name, value):
        set_array_dynamic_property(self.v, name, value)

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

    def __index__(self):
        return array_index(self.v)

    def __int__(self):
        return array_int(self.v)

    def __float__(self):
        return array_float(self.v)

    def __complex__(self):
        return array_complex(self.v)

    def __len__(self):
        return self.v.get_dim_size()

    def __nonzero__(self):
        return array_nonzero(self.v)

    def __repr__(self):
        return str(<char *>array_repr(self.v).c_str())

    def __setitem__(self, x, y):
        array_setitem(self.v, x, y)

    def __str__(self):
        try:
            return array_str(self.v)
        except TypeError:
            return self.__repr__()

    def __unicode__(self):
        return array_unicode(self.v)

    def __add__(lhs, rhs):
        cdef array res = array()
        res.v = array_add(asarray(lhs).v, asarray(rhs).v)
        return res

    def __sub__(lhs, rhs):
        cdef array res = array()
        res.v = array_subtract(asarray(lhs).v, asarray(rhs).v)
        return res

    def __mul__(lhs, rhs):
        cdef array res = array()
        res.v = array_multiply(asarray(lhs).v, asarray(rhs).v)
        return res

    def __div__(lhs, rhs):
        cdef array res = array()
        res.v = array_divide(asarray(lhs).v, asarray(rhs).v)
        return res

    def __truediv__(lhs, rhs):
        cdef array res = array()
        res.v = array_divide(asarray(lhs).v, asarray(rhs).v)
        return res

    def __richcmp__(a0, a1, int op):
        if op == Py_LT:
            return wrap(asarray(a0).v < asarray(a1).v)
        elif op == Py_LE:
            return wrap(asarray(a0).v <= asarray(a1).v)
        elif op == Py_EQ:
            return wrap(asarray(a0).v == asarray(a1).v)
        elif op == Py_NE:
            return wrap(asarray(a0).v != asarray(a1).v)
        elif op == Py_GE:
            return wrap(asarray(a0).v >= asarray(a1).v)
        elif op == Py_GT:
            return wrap(asarray(a0).v > asarray(a1).v)

    def __getbuffer__(array self, Py_buffer* buffer, int flags):
        # Docstring triggered Cython bug (fixed in master), so it's commented out
        #"""PEP 3118 buffer protocol"""
        array_getbuffer_pep3118(self, buffer, flags)

    def __releasebuffer__(array self, Py_buffer* buffer):
        # Docstring triggered Cython bug (fixed in master), so it's commented out
        #"""PEP 3118 buffer protocol"""
        array_releasebuffer_pep3118(self, buffer)

    def cast(self, tp):
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
        cdef array result = array()
        result.v = array_cast(self.v, type(tp).v)
        return result

    def eval(self, ectx=None):
        """
        a.eval(ectx=<default eval_context>)
        Returns a version of the dynd array with plain values,
        all expressions evaluated. This returns the original
        array back if it has no expression type.
        Parameters
        ----------
        ectx : nd.eval_context
            The evaluation context to use.
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
        cdef array result = array()
        result.v = array_eval(self.v, ectx)
        return result

    def sum(self, axis = None):
      from .. import nd

      if (axis is None):
        return nd.sum(self)

      return nd.sum(self, axes = [axis])

    def ucast(self, dtype, int replace_ndim=0):
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
        cdef array result = array()
        result.v = array_ucast(self.v, type(dtype).v, replace_ndim)
        return result

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
        result.v = self.v.view_scalars(type(dtp).v)
        return result

set_wrapper_type[_array](array)

cdef class array_callable:
    cdef array_callable_wrapper v

    def __call__(self, *args, **kwargs):
        return array_callable_call(self.v, args, kwargs)

init_w_array_callable_typeobject(array_callable)

cpdef array asarray(obj, access=None):
    """
    nd.asarray(obj, access=None)
    Constructs a dynd array from the object, taking a view
    if possible, otherwise making a copy.
    Parameters
    ----------
    obj : object
        The object which is to be converted into a dynd array,
        as a view if possible, otherwise a copy.
    access : 'readwrite'/'rw', 'readonly'/'r', 'immutable', optional
        If provided, the access flags the resulting array should
        satisfy. When a view can be taken, but these access flags
        cannot, a copy is made.
    """

    cdef array result = array()
    if isinstance(obj, tuple):
        obj = list(obj)
    result.v = array_asarray(obj, access)
    return result

from dynd.nd.callable cimport callable

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
    cdef type result = type()
    if isinstance(a, array):
        result.v = (<array> a).v.get_type()
    elif isinstance(a, callable):
        result.v = (<callable> a).v.get_array_type()

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
    cdef type result = type()
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
    return str(<char *>dynd_format_datashape(a.v).c_str())

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

def as_py(array n, tuple=False):
    """
    nd.as_py(n, tuple=False)
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
    cdef bint tup = tuple
    return array_as_py(n.v, tup != 0)

def as_numpy(array n, allow_copy=False):
    """
    nd.as_numpy(n, allow_copy=False)
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
    # TODO: Could also convert dynd types into numpy dtypes
    return array_as_numpy(n, bool(allow_copy))

def view(obj, type=None, access=None):
    """
    nd.view(obj, type=None, access=None)
    Constructs a dynd array which is a view of the data from
    `obj`. The `access` parameter can be used to require writable
    access for an output parameter, or to produce a read-only
    view of writable data.
    Parameters
    ----------
    obj : object
        A Python object which backs some array data, such as
        a dynd array, a numpy array, or an object supporting
        the Python buffer protocol.
    type : ndt.type, optional
        If provided, requests that the memory of ``obj`` be viewed
        as this type.
    access : 'readwrite'/'rw' or 'readonly'/'r', optional
        The access flags for the constructed array. Use 'readwrite'
        to require that the view be writable, and 'readonly' to
        provide a view of data to someone else without allowing
        writing.
    """
    cdef array result = array()
    result.v = array_view(obj, type, access)
    return result

def zeros(*args, **kwargs):
    """
    nd.zeros(dtype, *, access=None)
    nd.zeros(shape, dtype, *, access=None)
    nd.zeros(shape_0, shape_1, ..., shape_(n-1), dtype, *, access=None)
    Creates an array of zeros of the specified
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
    access : 'readwrite' or 'immutable', optional
        Specifies the access control of the resulting copy. Defaults
        to readwrite.
    """
    # Handle the keyword-only arguments
    access = kwargs.pop('access', None)
    if kwargs:
        msg = "nd.zeros() got an unexpected keyword argument '%s'"
        raise TypeError(msg % (kwargs.keys()[0]))

    cdef array result = array()
    largs = len(args)
    if largs  == 1:
        # Only the full type is provided
        result.v = array_zeros(type(args[0]).v, access)
    elif largs == 2:
        # The shape is a provided as a tuple (or single integer)
        result.v = array_zeros(args[0], type(args[1]).v, access)
    elif largs > 2:
        # The shape is expanded out in the arguments
        result.v = array_zeros(args[:-1], type(args[-1]).v, access)
    else:
        raise TypeError('nd.zeros() expected at least 1 positional argument, got 0')
    return result

def ones(*args, **kwargs):
    """
    nd.ones(type, *, access=None)
    nd.ones(shape, dtype, *, access=None)
    nd.ones(shape_0, shape_1, ..., shape_(n-1), dtype, *, access=None)
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
    access : 'readwrite' or 'immutable', optional
        Specifies the access control of the resulting copy. Defaults
        to readwrite.
    """
    # Handle the keyword-only arguments
    access = kwargs.pop('access', None)
    if kwargs:
        msg = "nd.ones() got an unexpected keyword argument '%s'"
        raise TypeError(msg % (kwargs.keys()[0]))

    cdef array result = array()
    largs = len(args)
    if largs  == 1:
        # Only the full type is provided
        result.v = array_ones(type(args[0]).v, access)
    elif largs == 2:
        # The shape is a provided as a tuple (or single integer)
        result.v = array_ones(args[0], type(args[1]).v, access)
    elif largs > 2:
        # The shape is expanded out in the arguments
        result.v = array_ones(args[:-1], type(args[-1]).v, access)
    else:
        raise TypeError('nd.ones() expected at least 1 positional argument, got 0')
    return result

def full(*args, **kwargs):
    """
    nd.full(type, *, value, access=None)
    nd.ones(shape, dtype, *, value, access=None)
    nd.ones(shape_0, shape_1, ..., shape_(n-1), dtype, *, value, access=None)
    Creates an array filled with the given value and
    of the specified type. Dimensions may be provided as
    integer positional parameters, a tuple of integers,
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
    access : 'readwrite' or 'immutable', optional
        Specifies the access control of the resulting copy. Defaults
        to readwrite.
    Examples
    --------
    >>> from dynd import nd, ndt
    >>> nd.full(2, 3, ndt.int32, value=123)
    nd.array([[123, 123, 123], [123, 123, 123]],
             type="2 * 3 * int32")
    >>> nd.full(2, 2, ndt.int32, value=[1, 5])
    nd.array([[1, 5], [1, 5]],
             type="2 * 2 * int32")
    >>> nd.full('3 * {x : int32, y : 2 * int16}', value=[1, [2, 3]], access='rw')
    nd.array([[1, [2, 3]], [1, [2, 3]], [1, [2, 3]]],
             type="3 * {x : int32, y : 2 * int16}")
    """
    # Handle the keyword-only arguments
    if 'value' not in kwargs:
        raise TypeError("nd.full() missing required " +
                    "keyword-only argument: 'value'")
    access = kwargs.pop('access', None)
    value = kwargs.pop('value', None)
    if kwargs:
        msg = "nd.full() got an unexpected keyword argument '%s'"
        raise TypeError(msg % (kwargs.keys()[0]))

    cdef array result = array()
    largs = len(args)
    if largs  == 1:
        # Only the full type is provided
        result.v = array_full(type(args[0]).v, value, access)
    elif largs == 2:
        # The shape is a provided as a tuple (or single integer)
        result.v = array_full(args[0], type(args[1]).v, value, access)
    elif largs > 2:
        # The shape is expanded out in the arguments
        result.v = array_full(args[:-1], type(args[-1]).v, value, access)
    else:
        raise TypeError('nd.full() expected at least 1 positional argument, got 0')
    return result

def empty(*args, **kwargs):
    """
    nd.empty(type, access=None)
    nd.empty(shape, dtype, access=None)
    nd.empty(shape_0, shape_1, ..., shape_(n-1), dtype, access=None)
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
    access : 'readwrite' or 'immutable', optional
        Specifies the access control of the resulting copy. Defaults
        to readwrite.
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
    # Handle the keyword-only arguments
    access = kwargs.pop('access', None)
    if kwargs:
        msg = "nd.empty() got an unexpected keyword argument '%s'"
        raise TypeError(msg % (kwargs.keys()[0]))

    cdef array result = array()
    largs = len(args)
    if largs  == 1:
        # Only the full type is provided
        result.v = array_empty(type(args[0]).v, access)
    elif largs == 2:
        # The shape is a provided as a tuple (or single integer)
        result.v = array_empty(args[0], type(args[1]).v, access)
    elif largs > 2:
        # The shape is expanded out in the arguments
        result.v = array_empty(args[:-1], type(args[-1]).v, access)
    else:
        raise TypeError('nd.empty() expected at least 1 positional argument, got 0')
    return result

def range(start=None, stop=None, step=None, dtype=None):
    """
    nd.range(stop, dtype=None)
    nd.range(start, stop, step=None, dtype=None)
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
            result.v = array_range(None, start, step, dtype)
        else:
            raise ValueError("No value provided for 'stop'")
    else:
        result.v = array_range(start, stop, step, dtype)
    return result

def linspace(start, stop, count=50, dtype=None):
    """
    nd.linspace(start, stop, count=50, dtype=None)
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
    result.v = array_linspace(start, stop, count, dtype)
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
        result.v = dynd_parse_json_type(type(tp).v, array(json).v, ectx)
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

from ..cpp.json_parser cimport parse as _parse

def parse(tp, obj):
    return wrap(_parse((<type> tp).v, str(obj)))

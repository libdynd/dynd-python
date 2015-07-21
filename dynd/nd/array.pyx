from dynd.ndt.type cimport type
from dynd.ndt import Unsupplied

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

    property shape:
        def __get__(self):
            return array_get_shape(self.v)

    def __getattr__(self, name):
        return get_array_dynamic_property(self.v, name)

    def __getitem__(self, x):
        cdef array result = array()
        result.v = array_getitem(self.v, x)
        return result

    def __len__(self):
        return self.v.get_dim_size()

    def __setitem__(self, x, y):
        array_setitem(self.v, x, y)

init_w_array_typeobject(array)

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
    result.v = array_asarray(obj, access)
    return result

def type_of(array a):
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
    result.v = a.v.get_type()
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
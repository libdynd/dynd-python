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

    def __getattr__(self, name):
        return get_array_dynamic_property(self.v, name)

init_w_array_typeobject(array)

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
from cpython.object cimport Py_EQ, Py_NE

cdef class type(object):
    """
    ndt.type(obj=None)

    Create a dynd type object.

    A dynd type object describes the dimensional
    structure and element type of a dynd array.

    Parameters
    ----------
    obj : string or other data type, optional
        A Blaze datashape string or a data type from another
        system such as NumPy or ctypes.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> ndt.type('int16')
    ndt.int16
    >>> ndt.type('5 * var * float32')
    ndt.type("5 * var * float32")
    >>> ndt.type('{x: float32, y: float32, z: float32}')
    ndt.type("{x : float32, y : float32, z : float32}")
    """

    def __cinit__(self, rep=None):
        if rep is not None:
            self.v = make__type_from_pyobject(rep)

    property shape:
        """
        tp.shape
        The shape of the array dimensions of the type. For
        dimensions whose size is unknown without additional
        array arrmeta or array data, a -1 is returned.
        """
        def __get__(self):
            return _type_get_shape(self.v)

    property dshape:
        """
        tp.dshape
        The blaze datashape of the dynd type, as a string.
        """
        def __get__(self):
            return str(<char *>dynd_format_datashape(self.v).c_str())

    property data_size:
        """
        tp.data_size
        The size, in bytes, of the data for an instance
        of this dynd type.
        None is returned if array arrmeta is required to
        fully specify it. For example, both the fixed and
        struct types require such arrmeta.
        """
        def __get__(self):
            cdef ssize_t result = self.v.get_data_size()
            if result > 0:
                return result
            else:
                return None

    property default_data_size:
        """
        tp.default_data_size
        The size, in bytes, of the data for a default-constructed
        instance of this dynd type.
        """
        def __get__(self):
            return self.v.get_default_data_size()

    property data_alignment:
        """
        tp.data_alignment
        The alignment, in bytes, of the data for an
        instance of this dynd type.
        Data for this dynd type must always be aligned
        according to this alignment, unaligned data
        requires an adapter transformation applied.
        """
        def __get__(self):
            return self.v.get_data_alignment()

    property arrmeta_size:
        """
        tp.arrmeta_size
        The size, in bytes, of the arrmeta for
        this dynd type.
        """
        def __get__(self):
            return self.v.get_arrmeta_size()

    property kind:
        """
        tp.kind
        The kind of this dynd type, as a string.
        Example kinds are 'bool', 'int', 'uint',
        'real', 'complex', 'string', 'uniform_array',
        'expression'.
        """
        def __get__(self):
            return _type_get_kind(self.v)

    property type_id:
        """
        tp.type_id
        The type id of this dynd type, as a string.
        Example type ids are 'bool', 'int8', 'uint32',
        'float64', 'complex_float32', 'string', 'byteswap'.
        """
        def __get__(self):
            return _type_get_type_id(self.v)

    def __getattr__(self, name):
        return get__type_dynamic_property(self.v, name)

    def __str__(self):
        return str(<char *>_type_str(self.v).c_str())

    def __repr__(self):
        return str(<char *>_type_repr(self.v).c_str())

    def __richcmp__(lhs, rhs, int op):
        if op == Py_EQ:
            if isinstance(lhs, type) and isinstance(rhs, type):
                return (<type>lhs).v == (<type>rhs).v
            else:
                return False
        elif op == Py_NE:
            if isinstance(lhs, type) and isinstance(rhs, type):
                return (<type>lhs).v != (<type>rhs).v
            else:
                return False
        return NotImplemented

    def match(self, rhs):
        """
        tp.match(candidate)
        Returns True if the candidate type ``candidate`` matches the possibly
        symbolic pattern type ``tp``, False otherwise.
        Examples
        --------
        >>> from dynd import nd, ndt
        >>> ndt.type("T").match(ndt.int32)
        True
        >>> ndt.type("Dim * T").match(ndt.int32)
        False
        >>> ndt.type("M * {x: ?T, y: T}").match("10 * {x : ?int32, y : int32}")
        True
        >>> ndt.type("M * {x: ?T, y: T}").match("10 * {x : ?int32, y : ?int32}")
        False
        """
        return self.v.match(type(rhs).v)


init_w_type_typeobject(type)

class UnsuppliedType(object):
    pass

Unsupplied = UnsuppliedType()

def make_fixed_bytes(int data_size, int data_alignment=1):
    """
    ndt.make_fixed_bytes(data_size, data_alignment=1)
    Constructs a bytes type with the specified data size and alignment.
    Parameters
    ----------
    data_size : int
        The number of bytes of one instance of this dynd type.
    data_alignment : int, optional
        The required alignment of instances of this dynd type.
        This value must be a small power of two. Default: 1.
    Examples
    --------
    >>> from dynd import nd, ndt
    >>> ndt.make_fixed_bytes(4)
    ndt.type("bytes[4]")
    >>> ndt.make_fixed_bytes(6, 2)
    ndt.type("bytes[6, align=2]")
    """
    cdef type result = type()
    result.v = dynd_make_fixed_bytes_type(data_size, data_alignment)
    return result

def make_fixed_dim(shape, element_tp):
    """
    ndt.make_fixed_dim(shape, element_tp)
    Constructs a fixed_dim type of the given shape.
    Parameters
    ----------
    shape : tuple of int
        The multi-dimensional shape of the resulting fixed array type.
    element_tp : dynd type
        The type of each element in the resulting array type.
    Examples
    --------
    >>> from dynd import nd, ndt
    >>> ndt.make_fixed_dim(5, ndt.int32)
    ndt.type("5 * int32")
    >>> ndt.make_fixed_dim((3,5), ndt.int32)
    ndt.type("3 * 5 * int32")
    """
    cdef type result = type()
    result.v = dynd_make_fixed_dim_type(shape, type(element_tp).v)
    return result

def make_fixed_string(int size, encoding=None):
    """
    ndt.make_fixed_string(size, encoding='utf_8')
    Constructs a fixed-size string type with a specified encoding,
    whose size is the specified number of base units for the encoding.
    Parameters
    ----------
    size : int
        The number of base encoding units in the data. For example,
        for UTF-8, a size of 10 means 10 bytes. For UTF-16, a size
        of 10 means 20 bytes.
    encoding : string, optional
        The encoding used for storing unicode code points. Supported
        values are 'ascii', 'utf_8', 'utf_16', 'utf_32', 'ucs_2'.
        Default: 'utf_8'.
    Examples
    --------
    >>> from dynd import nd, ndt
    >>> ndt.make_fixed_string(10)
    ndt.type("string[10]")
    >>> ndt.make_fixed_string(10, 'utf_32')
    ndt.type("string[10,'utf32']")
    """
    cdef type result = type()
    result.v = dynd_make_fixed_string_type(size, encoding)
    return result

def make_string(encoding=None):
    """
    ndt.make_string(encoding='utf_8')
    Constructs a variable-sized string dynd type
    with the specified encoding.
    Parameters
    ----------
    encoding : string, optional
        The encoding used for storing unicode code points. Supported
        values are 'ascii', 'utf_8', 'utf_16', 'utf_32', 'ucs_2'.
        Default: 'utf_8'.
    Examples
    --------
    >>> from dynd import nd, ndt
    >>> ndt.make_string()
    ndt.string
    >>> ndt.make_string('utf_16')
    ndt.type("string['utf16']")
    """
    cdef type result = type()
    result.v = dynd_make_string_type(encoding)
    return result

def make_struct(field_types, field_names):
    """
    ndt.make_struct(field_types, field_names)
    Constructs a struct dynd type, which has fields with a flexible
    per-array layout.
    If a subset of fields from a fixed_struct are taken,
    the result is a struct, with the layout specified
    in the dynd array's arrmeta.
    Parameters
    ----------
    field_types : list of dynd types
        A list of types, one for each field.
    field_names : list of strings
        A list of names, one for each field, corresponding to 'field_types'.
    Examples
    --------
    >>> from dynd import nd, ndt
    >>> ndt.make_struct([ndt.int32, ndt.float64], ['x', 'y'])
    ndt.type("{x : int32, y : float64}")
    """
    cdef type result = type()
    result.v = dynd_make_struct_type(field_types, field_names)
    return result

def make_fixed_dim_kind(element_tp, ndim=None):
    """
    ndt.make_fixed_dim_kind(element_tp, ndim=1)
    Constructs an array dynd type with one or more symbolic fixed
    dimensions. A single fixed_dim_kind dynd type corresponds
    to one dimension, so when ndim > 1, multiple fixed_dim_kind
    dimensions are created.
    Parameters
    ----------
    element_tp : dynd type
        The type of one element in the symbolic array.
    ndim : int
        The number of fixed_dim_kind dimensions to create.
    Examples
    --------
    >>> from dynd import nd, ndt
    >>> ndt.make_fixed_dim_kind(ndt.int32)
    ndt.type("Fixed * int32")
    >>> ndt.make_fixed_dim_kind(ndt.int32, 3)
    ndt.type("Fixed * Fixed * Fixed * int32")
    """
    cdef type result = type()
    if (ndim is None):
        result.v = dynd_make_fixed_dim_kind_type(type(element_tp).v)
    else:
        result.v = dynd_make_fixed_dim_kind_type(type(element_tp).v, int(ndim))
    return result


def make_var_dim(element_tp):
    """
    ndt.make_fixed_dim(element_tp)
    Constructs a var_dim type.
    Parameters
    ----------
    element_tp : dynd type
        The type of each element in the resulting array type.
    Examples
    --------
    >>> from dynd import nd, ndt
    >>> ndt.make_var_dim(ndt.float32)
    ndt.type("var * float32")
    """
    cdef type result = type()
    result.v = dynd_make_var_dim_type(type(element_tp).v)
    return result
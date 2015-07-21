# Helper for cases where we can't use None for a missing argument default
class UnsuppliedType(object):
    pass
Unsupplied = UnsuppliedType()

from dynd.config cimport w_type

def make_byteswap(builtin_type, operand_type=None):
    """
    ndt.make_byteswap(builtin_type, operand_type=None)

    Constructs a byteswap type from a builtin one, with an
    optional expression type to chain in as the operand.

    Parameters
    ----------
    builtin_type : dynd type
        The builtin dynd type (like ndt.int16, ndt.float64) to
        which to apply the byte swap operation.
    operand_type: dynd type, optional
        An expression dynd type whose value type is a fixed bytes
        dynd type with the same data size and alignment as
        'builtin_type'.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> ndt.make_byteswap(ndt.int16)
    ndt.type("byteswap[int16]")
    """
    cdef w_type result = w_type()
    if operand_type is None:
        result.v = dynd_make_byteswap_type(w_type(builtin_type).v)
    else:
        result.v = dynd_make_byteswap_type(w_type(builtin_type).v, w_type(operand_type).v)
    return result

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
    cdef w_type result = w_type()
    result.v = dynd_make_fixed_bytes_type(data_size, data_alignment)
    return result

def make_convert(to_tp, from_tp):
    """
    ndt.make_convert(to_tp, from_tp)

    Constructs an expression type which converts from one
    dynd type to another.

    Parameters
    ----------
    to_tp : dynd type
        The dynd type being converted to. This is the 'value_type'
        of the resulting expression dynd type.
    from_tp : dynd type
        The dynd type being converted from. This is the 'operand_type'
        of the resulting expression dynd type.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> ndt.make_convert(ndt.int16, ndt.float32)
    ndt.type("convert[to=int16, from=float32]")
    """
    cdef w_type result = w_type()
    result.v = dynd_make_convert_type(w_type(to_tp).v, w_type(from_tp).v)
    return result

def make_view(value_type, operand_type):
    """
    ndt.make_view(value_type, operand_type)

    Constructs an expression type which views the bytes of
    one type as another.

    Parameters
    ----------
    value_type : dynd type
        The dynd type to interpret the bytes as. This is the 'value_type'
        of the resulting expression dynd type.
    operand_type : dynd type
        The dynd type the memory originally was. This is the 'operand_type'
        of the resulting expression dynd type.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> ndt.make_view(ndt.int32, ndt.uint32)
    ndt.type("view[as=int32, original=uint32]")
    """
    cdef w_type result = w_type()
    result.v = dynd_make_view_type(w_type(value_type).v, w_type(operand_type).v)
    return result

def make_unaligned(aligned_tp):
    """
    ndt.make_unaligned(aligned_tp)

    Constructs a type with alignment of 1 from the given type.
    If the type already has alignment 1, just returns it.

    Parameters
    ----------
    aligned_tp : dynd type
        The dynd type which should be viewed on data that is
        not properly aligned.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> ndt.make_unaligned(ndt.int32)
    ndt.type("unaligned[int32]")
    >>> ndt.make_unaligned(ndt.uint8)
    ndt.uint8
    """
    cdef w_type result = w_type()
    result.v = dynd_make_unaligned_type(w_type(aligned_tp).v)
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
    cdef w_type result = w_type()
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
    cdef w_type result = w_type()
    result.v = dynd_make_string_type(encoding)
    return result

def make_bytes(size_t alignment=1):
    """
    ndt.make_bytes(alignment=1)

    Constructs a variable-sized bytes dynd type
    with the specified alignment.

    Parameters
    ----------
    alignment : int, optional
        The byte alignment of the raw binary data.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> ndt.make_bytes()
    ndt.bytes
    >>> ndt.make_bytes(4)
    ndt.type("bytes[align=4]")
    """
    cdef w_type result = w_type()
    result.v = dynd_make_bytes_type(alignment)
    return result

def make_pointer(target_tp):
    """
    ndt.make_pointer(target_tp)

    Constructs a dynd type which is a pointer to the target type.

    Parameters
    ----------
    target_tp : dynd type
        The type that the pointer points to. This is similar to
        the '*' in C/C++ type declarations.
    """
    cdef w_type result = w_type()
    result.v = dynd_make_pointer_type(w_type(target_tp).v)
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
    cdef w_type result = w_type()
    if (ndim is None):
        result.v = dynd_make_fixed_dim_kind_type(w_type(element_tp).v)
    else:
        result.v = dynd_make_fixed_dim_kind_type(w_type(element_tp).v, int(ndim))
    return result

def make_pow_dimsym(base_tp, exponent, element_tp):
    """
    ndt.make_fixed_dim_sym(base_tp, exponent, element_tp)

    Constructs an array dynd type with a symbolic dimensional
    power type, where the base type is raised to the power
    of the exponent.

    Parameters
    ----------
    base_tp : dynd type
        The type of of the base of the dimensional power.
        This should be of the form ``<dim> * void``.
    exponent : str
        The name of the typevar for the exponent.
    element_tp : dynd type
        The type of one element after the dimensional power.

    Examples
    --------
    >>> from dynd import nd, ndt

    """
    cdef w_type result = w_type()
    result.v = dynd_make_pow_dimsym_type(w_type(base_tp).v, exponent,
                                         w_type(element_tp).v)
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
    cdef w_type result = w_type()
    result.v = dynd_make_fixed_dim_type(shape, w_type(element_tp).v)
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
    cdef w_type result = w_type()
    result.v = dynd_make_struct_type(field_types, field_names)
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
    cdef w_type result = w_type()
    result.v = dynd_make_var_dim_type(w_type(element_tp).v)
    return result

def make_property(operand_tp, name):
    """
    ndt.make_property(operand_tp, name):

    Constructs a property type from `operand_tp`, with name `name`. For
    example the "real" property from a "complex[float64]" type.

    The resulting type has an operand type matching `operand_tp`, and
    a value type derived from the property.

    Parameters
    ----------
    operand_tp : dynd type
        The type from which to construct the property type. This will be
        the operand type of the resulting property type.
    name : str
        The name of the property
    """
    cdef w_type result = w_type()
    n = str(name)
    result.v = dynd_make_property_type(w_type(operand_tp).v, string(<const char *>n))
    return result

def make_reversed_property(value_tp, operand_tp, name):
    """
    ndt.make_reversed_property(value_tp, operand_tp, name):

    Constructs a reversed property type from `tp`, with name `name`. For
    example the "struct" property from a "date" type.

    The resulting type has an operand type derived from the property,
    and a value type equal to `tp`.

    Parameters
    ----------
    value_tp : dynd type
        A type whose value type must be equal to the value type of the selected
        property.
    operand_tp : dynd type
        The type from which the property is taken. This will be the value
        type of the result.
    name : str
        The name of the property.
    """
    cdef w_type result = w_type()
    n = str(name)
    result.v = dynd_make_reversed_property_type(w_type(value_tp).v,
                                                w_type(operand_tp).v, string(<const char *>n))
    return result

from dynd.config cimport _array, w_array

def make_categorical(values):
    """
    ndt.make_categorical(values)

    Constructs a categorical dynd type with the
    specified values as its categories.

    Instances of the resulting type are integers, consisting
    of indices into this values array. The size of the values
    array controls what kind of integers are used, if there
    are 256 or fewer categories, a uint8 is used, if 65536 or
    fewer, a uint16 is used, and otherwise a uint32 is used.

    Parameters
    ----------
    values : one-dimensional array
        This is an array of the values that become the categories
        of the resulting type. The values must be unique.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> ndt.make_categorical(['sunny', 'rainy', 'cloudy', 'stormy'])
    ndt.type("categorical[string, [\"sunny\", \"rainy\", \"cloudy\", \"stormy\"]]")
    """
    cdef w_type result = w_type()
    result.v = dynd_make_categorical_type(w_array(values).v)
    return result

def factor_categorical(values):
    """
    ndt.factor_categorical(values)

    Constructs a categorical dynd type with the
    unique sorted subset of the values as its
    categories.

    Parameters
    ----------
    values : one-dimensional dynd array
        This is an array of the values that are sorted, with
        duplicates removed, to produce the categories of
        the resulting dynd type.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> ndt.factor_categorical(['M', 'M', 'F', 'F', 'M', 'F', 'M'])
    ndt.type("categorical[string, [\"F\", \"M\"]]")
    """
    cdef w_type result = w_type()
    result.v = dynd_factor_categorical_type(w_array(values).v)
    return result

def extract_dtype(dt, size_t include_ndim=0):
    """
    extract_dtype(dt, include_ndim=0)

    Extracts the uniform type from the provided
    dynd type. If `keep_ndim` is positive, that
    many array dimensions are kept in the result.

    Parameters
    ----------
    dt : dynd type
        The dtype whose dtype is to be extracted.
    keep_ndim : integer, optional
        If positive, this is the number of array dimensions
        which are extracted in the dtype for replacement.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> d = ndt.type('3 * var * int32')
    >>> ndt.extract_dtype(d)
    ndt.int32
    >>> ndt.extract_dtype(d, 1)
    ndt.type("var * int32")
    """
    cdef w_type result = w_type()
    result.v = w_type(dt).v.get_dtype(include_ndim)
    return result

def replace_dtype(w_type dt, replacement_dt, size_t replace_ndim=0):
    """
    replace_dtype(dt, replacement_dt, replace_ndim=0)

    Replaces the dtype with the replacement.
    If `replace_ndim` is positive, that number of uniform
    dimensions are replaced as well.

    Parameters
    ----------
    dt : dynd type
        The dtype whose dtype is to be replaced.
    replacement_dt : dynd type
        The replacement dynd type.
    replace_ndim : integer, optional
        If positive, this is the number of array dimensions
        which are included in the dtype for replacement.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> d = ndt.type('3 * var * int32')
    >>> ndt.replace_dtype(d, 'Fixed * float64')
    ndt.type("3 * var * Fixed * float64")
    >>> ndt.replace_dtype(d, '{x: int32, y:int32}', 1)
    ndt.type("3 * {x : int32, y : int32}")
    """
    cdef w_type result = w_type()
    result.v = dt.v.with_replaced_dtype(w_type(replacement_dt).v, replace_ndim)
    return result
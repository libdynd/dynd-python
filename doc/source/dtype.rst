DyND DTypes
===========

DyND has a preliminary set of dtypes, similar to the ones
in NumPy.

Currently, dtypes are printed using angle brackets <> for the
parameters determining the nature of the dtype. This is a temporary
convention, and will be replaced when a reasonable, extensible syntax
has been determined. The time frame for this design is after the dtype
system has a fairly high number of dtypes with complicated parmeters.

Primitive DTypes
----------------

To start are the primitive numeric dtypes with a fixed size.

============  =============== =====================================
DType          Size (bytes)    Notes
============  =============== =====================================
ndt.bool       1               True (value 0) or False (value 1)
ndt.int8       1               8 bit signed integer
ndt.int16      2               16 bit signed integer
ndt.int32      4               32 bit signed integer
ndt.int64      8               64 bit signed integer
ndt.uint8      1               8 bit unsigned integer
ndt.uint16     2               16 bit unsigned integer
ndt.uint32     4               32 bit unsigned integer
ndt.uint64     8               64 bit unsigned integer
ndt.float32    4               32-bit IEEE floating point
ndt.float64    8               64-bit IEEE floating point
ndt.cfloat32   8               complex made of 32-bit IEEE floats
ndt.cfloat64   16              complex made of 64-bit IEEE floats
============  =============== =====================================

Bytes DType
-----------

There is a fixed size bytes dtype, which has a configurable size
and alignment. For example, one could make a bytes dtype with
size 8 and alignment 4 to match the storage properties of complex
float32.

.. code-block:: python

    >>> ndt.make_fixedbytes_dtype(8, 4)
    nd.dtype('fixedbytes<8,4>')

Unaligned DType
---------------

Any memory containing data for a dtype must always obey the dtype's
alignment. This is different from Numpy, where data may have different
alignment, and NumPy checks whether there is alignment and
inserts buffering where necessary. To deal with unaligned data,
a dtype which adapts it into aligned storage must be used.

.. code-block:: python

    >>> ndt.make_unaligned_dtype(ndt.float64)
    nd.dtype('unaligned<float64>')

Byteswap DType
--------------

When data has the wrong endianness for the platform, it must be
byteswapped before it can be used. The `byteswap` adapter dtype
plays this role.

.. code-block:: python

    >>> ndt.make_byteswap_dtype(ndt.int32)
    nd.dtype('byteswap<int32>')

Convert DType
-------------

Conversion between dtypes is handled by the `convert` dtype. The
error mode used for the conversion can be controlled with an extra
parameter.

.. code-block:: python

    >>> ndt.make_convert_dtype(ndt.int32, ndt.float64)
    nd.dtype('convert<to=int32, from=float64>')

    >>> ndt.make_convert_dtype(ndt.int32, ndt.float64, errmode='overflow')
    nd.dtype('convert<to=int32, from=float64, errmode=overflow>')


String DTypes
-------------

There are presently two different kinds of string dtypes. There are
fixed size strings, similar to those in NumPy, where the string data
is null-terminated within a buffer of particular size.

There are also variable size strings, whose data is in a separate
memory block, reference counted at the memory block level (blockref).
These strings cannot be resized in place, but are designed to be used
in a functional style where results are created once, not repeatedly
mutated in place.

Also planned are strings with full dynamic behavior, where each string
manages its own memory allocation. These are not implemented yet.

To create string dtypes, use the `ndt.make_string_dtype` and
`ndt.make_fixedstring_dtype`. There is a default string type
`ndt.string`, which is UTF-8.

.. code-block:: python

    >>> ndt.make_string_dtype('ascii')
    nd.dtype('string<ascii>')

    >>> ndt.make_fixedstring_dtype(16, 'utf_32')
    nd.dtype("string<16,'utf-32'>")

When creating ndarray objects from Python lists, blockref strings
are used by default.

.. code-block:: python

    >>> nd.ndobject(['abcdefg', u'안녕', u'Testing'])
    nd.ndobject(["abcdefg", "\uc548\ub155", "Testing"], string<ucs_2>)

Categorical DType
-----------------

There is a preliminary categorical dtype, used by the `nd.groupby`
function.

.. code-block:: python

    >>> groups = nd.ndobject(['a', 'b', 'c'],
                     udtype=ndt.make_fixedstring_dtype(1, 'ascii'))
    >>> ndt.make_categorical_dtype(groups)
    nd.dtype('categorical<string<1,'ascii'>, ["a", "b", "c"]>')

Pointer DType
-------------

This dtype presently exists to help with `ctypes` function pointer
interoperability, but eventually will behave in a blockref manner,
similar to the blockref string dtype.

.. code-block:: python

    >>> ndt.make_pointer_dtype(ndt.cfloat32)
    nd.dtype('pointer<complex<float32>>')


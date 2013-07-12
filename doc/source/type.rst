DyND Types
==========

DyND has a preliminary set of types, similar to the ones
in NumPy.

Currently, types are printed using angle brackets <> for the
parameters determining the nature of the type. This is a temporary
convention, and will be replaced when a reasonable, extensible syntax
has been determined. The time frame for this design is after the type
system has a fairly high number of types with complicated parmeters.

Primitive Types
----------------

To start are the primitive numeric types with a fixed size.

============  =============== =====================================
Type           Size (bytes)    Notes
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

Bytes Type
----------

There is a fixed size bytes type, which has a configurable size
and alignment. For example, one could make a bytes type with
size 8 and alignment 4 to match the storage properties of complex
float32.

.. code-block:: python

    >>> ndt.make_fixedbytes(8, 4)
    ndt.type('fixedbytes<8,4>')

Unaligned Type
--------------

Any memory containing data for a type must always obey the type's
alignment. This is different from Numpy, where data may have different
alignment, and NumPy checks whether there is alignment and
inserts buffering where necessary. To deal with unaligned data,
a type which adapts it into aligned storage must be used.

.. code-block:: python

    >>> ndt.make_unaligned(ndt.float64)
    ndt.type('unaligned<float64>')

Byteswap Type
-------------

When data has the wrong endianness for the platform, it must be
byteswapped before it can be used. The `byteswap` adapter type
plays this role.

.. code-block:: python

    >>> ndt.make_byteswap(ndt.int32)
    ndt.type('byteswap<int32>')

Convert Type
------------

Conversion between types is handled by the `convert` type. The
error mode used for the conversion can be controlled with an extra
parameter.

.. code-block:: python

    >>> ndt.make_convert(ndt.int32, ndt.float64)
    ndt.type('convert<to=int32, from=float64>')

    >>> ndt.make_convert(ndt.int32, ndt.float64, errmode='overflow')
    ndt.type('convert<to=int32, from=float64, errmode=overflow>')


String Types
------------

There are presently two different kinds of string types. There are
fixed size strings, similar to those in NumPy, where the string data
is null-terminated within a buffer of particular size.

There are also variable size strings, whose data is in a separate
memory block, reference counted at the memory block level (blockref).
These strings cannot be resized in place, but are designed to be used
in a functional style where results are created once, not repeatedly
mutated in place.

Also planned are strings with full dynamic behavior, where each string
manages its own memory allocation. These are not implemented yet.

To create string types, use the `ndt.make_string` and
`ndt.make_fixedstring`. There is a default string type
`ndt.string`, which is UTF-8.

.. code-block:: python

    >>> ndt.make_string('ascii')
    ndt.type('string<ascii>')

    >>> ndt.make_fixedstring(16, 'utf_32')
    ndt.type("string<16,'utf-32'>")

When creating ndarray objects from Python lists, blockref strings
are used by default.

.. code-block:: python

    >>> nd.array(['abcdefg', u'안녕', u'Testing'])
    nd.array(["abcdefg", "\uc548\ub155", "Testing"], string<ucs_2>)

Categorical Type
----------------

There is a preliminary categorical type, used by the `nd.groupby`
function.

.. code-block:: python

    >>> groups = nd.array(['a', 'b', 'c'],
                     udtype=ndt.make_fixedstring(1, 'ascii'))
    >>> ndt.make_categorical(groups)
    ndt.type('categorical<string<1,'ascii'>, ["a", "b", "c"]>')

Pointer Type
------------

This type presently exists to help with `ctypes` function pointer
interoperability, but eventually will behave in a blockref manner,
similar to the blockref string type.

.. code-block:: python

    >>> ndt.make_pointer(ndt.cfloat32)
    ndt.type('pointer<complex<float32>>')


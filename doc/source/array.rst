DyND Array
==========

In this preview release of DyND, there is an `nd.array` object
which hold a multi-dimensional array of elements of a specific
`ndt.type`.

One of the important features of this library is interoperability with
NumPy that is as seamless as possible. When a dtype is compatible across
both systems, arrays from NumPy and DyND will implicitly move
between the libraries.

Importing DyND
--------------

The standard way to import DyND is with the command::

    >>> from dynd import nd, ndt

Constructing From Python Scalars
--------------------------------

Currently, scalars are always zero-dimensional arrays in the system.
This is likely to change in a future preview release.

There's a difference with NumPy in the default treatment of integers.
In NumPy, the C/C++ ``long`` type is used, which is 32-bits on 32-bit
platforms and on 64-bit Windows, and which is 64-bits on Linux and OS X.
In DyND if the Python integer fits in 32-bits, it will always
be initialized by default as a 32-bit integer.

.. code-block:: python

    >>> nd.array(0)
    nd.array(0, type="int32")

    >>> nd.array(3.14)
    nd.array(3.14, type="float64")

    >>> nd.array(1 + 1j)
    nd.array((1,1), type="complex[float64]")

Strings default to `blockref` strings, which are variable-sized strings.
The support for them is still preliminary, but some basic functionality
like converting between different unicode encodings is implemented.

.. code-block:: python

    >>> nd.array('testing')
    nd.array("testing", type="string")

Constructing from Python Lists
------------------------------

Similar to in NumPy, arrays can be constructed from lists of
objects. This code does not try to be as clever as NumPy, and
will fail if something is inconsistent in the input data.

.. code-block:: python

    >>> nd.array([True, False])
    nd.array([true, false], type="strided * bool")

    >>> nd.array([1,2,3])
    nd.array([1, 2, 3], type="strided * int32")

    >>> nd.array([[1.0,0],[0,1.0]])
    nd.array([[1, 0], [0, 1]], type="strided * strided * float64")

    >>> nd.array(["testing", "one", u"two", "three"])
    nd.array(["testing", "one", "two", "three"], type="strided * string")

Converting to Python Types
--------------------------

To convert back into native Python objects, there is an ``nd.as_py(a)``
function.

.. code-block:: python

    >> x = nd.array([True, False])
    >> nd.as_py(x)
    [True, False]

    >> x = nd.array("testing")
    >> nd.as_py(x)
    u'testing'

Constructing from NumPy Scalars
-------------------------------

Numpy scalars are also supported as input, and the dtype is preserved
in the conversion.

.. code-block:: python

    >>> import numpy as np

    >>> x = np.bool_(False)
    >>> nd.array(x)
    nd.array(false, type="bool")

    >>> x = np.int16(1000)
    >>> nd.array(x)
    nd.array(1000, type="int16")

    >>> x = np.complex128(3.1)
    >>> nd.array(x)
    nd.array((3.1,0), type="complex[float64]")

Constructing from NumPy Arrays
------------------------------

When the dtype is supported by DyND, NumPy arrays can
be converted into DyND arrays. When using the `nd.array` constructor,
a new array is created, but there is also `nd.asarray` which creates
a view when possible, and `nd.view` which always creates a view and
raises if that is not possible.

.. code-block:: python

    >>> x = np.arange(6.).reshape(3,2)
    >>> nd.array(x)
    nd.array([[0, 1], [2, 3], [4, 5]], type="strided * strided * float64")
    >>> nd.asarray(x)
    nd.array([[0, 1], [2, 3], [4, 5]], type="strided * strided * float64")
    >>> nd.view(x)
    nd.array([[0, 1], [2, 3], [4, 5]], type="strided * strided * float64")

    >>> x = np.array(['testing', 'one', 'two', 'three'])
    >>> nd.asarray(x)
    nd.array(["testing", "one", "two", "three"], type="strided * string[7,'utf32']")


Converting to NumPy Arrays
--------------------------

To support naturally feeding data into NumPy operations, the
NumPy array interface is used via the C struct PyArrayInterface.
This means NumPy operations will work on arrays with compatible
dtypes.

.. code-block:: python

    >>> x = nd.array([1, 2, 3.5])
    >>> np.square(x)
    array([  1.  ,   4.  ,  12.25])

There are some cases where a DyND array will not seamlessly convert
into a NumPy array. The behavior of NumPy is usually to ignore errors
that occur, and switch to an "object" array instead.

.. code-block:: python

    >>> x = nd.array([1, 2, 3]).ucast(ndt.float32)
    >>> x
    nd.array([1, 2, 3], type="strided * convert[to=float32, from=int32]")
    >>> np.array(x)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: expected a readable buffer object
    >>> nd.as_numpy(x)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "_pydynd.pyx", line 1406, in _pydynd.as_numpy (_pydynd.cxx:9568)
    RuntimeError: cannot view dynd array with dtype strided_dim<convert<to=float32,
    from=int32>> as numpy without making a copy
    >>> nd.as_numpy(x, allow_copy=True)
    array([ 1.,  2.,  3.], dtype=float32)
        

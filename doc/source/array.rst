DyND Array
==========

In this preview release of dynd, there is an `array` object
which hold a multi-dimensional array of elements of a specific `type`.

One of the important features of this library is interoperability with
numpy that is as seamless as possible. When a dtype is compatible across
both systems, arrays from numpy and dynd will implicitly move
between the libraries.

Importing DyND
--------------

The standard way to import dynd is with the command::

    >>> from dynd import nd, ndt

Constructing From Python Scalars
--------------------------------

Currently, scalars are always zero-dimensional arrays in the system.
This is likely to change in a future preview release.

There's a difference with numpy in the default treatment of integers.
In numpy, the C/C++ ``long`` type is used, which is 32-bits on 32-bit
platforms and on 64-bit Windows, and which is 64-bits on Linux and OS X.
In dynd if the Python integer fits in 32-bits, it will always
be initialized by default as a 32-bit integer.

.. code-block:: python

    >>> nd.array(0)
    nd.array(0, int32)

    >>> nd.array(3.14)
    nd.array(3.14, float64)

    >>> nd.array(1 + 1j)
    nd.array((1,1), complex<float64>)

Strings default to `blockref` strings, which are variable-sized strings.
The support for them is still preliminary, but some basic functionality
like converting between different unicode encodings is implemented.

One thing to note is that strings and unicode scalars are imported as
a view into the Python object's data. Because the Python object is immutable,
the array also is flagged as immutable.

.. code-block:: python

    >>> nd.array('testing')
    nd.array("testing", string<ascii>)

    >>> nd.array(u'testing')
    nd.array("testing", string<ucs_2>)

Constructing from Python Lists
------------------------------

Similar to in numpy, arrays can be constructed from lists of
objects. This code does not try to be as clever as numpy, and
will fail if something is inconsistent in the input data.

.. code-block:: python

    >>> nd.array([True, False])
    nd.array([true, false], bool)

    >>> nd.array([1,2,3])
    nd.array([1, 2, 3], int32)

    >>> nd.array([[1.0,0],[0,1.0]])
    nd.array([[1, 0], [0, 1]], float64)

    >>> nd.array(["testing", "one", u"two", "three"])
    nd.array(["testing", "one", "two", "three"], string<ucs_2>)

Converting to Python Types
--------------------------

To convert back into native Python objects, there is an ``as_py()``
function.

.. code-block:: python

    >> x = nd.array([True, False])
    >> x.as_py()
    [True, False]

    >> x = nd.array("testing")
    >> x.as_py()
    u'testing'

Constructing from NumPy Scalars
-------------------------------

Numpy scalars are also supported as input, and the dtype is preserved
in the conversion.

.. code-block:: python

    >>> x = np.bool_(False)
    >>> nd.array(x)
    nd.array(false, bool)

    >>> x = np.int16(1000)
    >>> nd.array(x)
    nd.array(1000, int16)

    >>> x = np.complex128(3.1)
    >>> nd.array(x)
    nd.array((3.1,0), complex<float64>)

Constructing from NumPy Arrays
------------------------------

When the dtype is supported by dynd, numpy arrays can
be converted into dynd arrays. The resulting array points at the same
data the numpy array used.

.. code-block:: python

    >>> x = np.arange(6.).reshape(3,2)
    >>> nd.array(x)
    nd.array([[0, 1], [2, 3], [4, 5]], float64)

    >>> x = np.array(['testing', 'one', 'two', 'three'])
    >>> nd.array(x)
    nd.array(["testing", "one", "two", "three"], fixedstring<ascii,7>)


Converting to NumPy Arrays
--------------------------

To support naturally feeding data into numpy operations, the
numpy array interface is used via the C struct PyArrayInterface.
This means numpy operations will work on arrays with compatible
dtypes.

.. code-block:: python

    >>> x = nd.array([1, 2, 3.5])
    >>> np.square(x)
    array([  1.  ,   4.  ,  12.25])


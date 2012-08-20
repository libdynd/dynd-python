Blaze-Local NDArray
===================

In this preview release of blaze-local, there is an `ndarray` object
which hold a multi-dimensional array of elements of a specific `dtype`.
Future versions of this library will likely explore some different
ways of organizing the data model, but something like the ndarray
will stay.

One of the important features of this library is interoperability with
Numpy that is as seamless as possible. When a dtype is compatible across
both systems, ndarrays from Numpy and blaze-local will implicitly move
between the libraries.

Constructing From Python Scalars
--------------------------------

Currently, scalars are always zero-dimensional arrays in the system.
This is likely to change in a future preview release.

There's a difference with Numpy in the default treatment of integers.
In Numpy, the C/C++ `long` type is used, which is 32-bits on 32-bit
platforms and on 64-bit Windows, and which is 64-bits on Linux and OS X.
If the Python integer fits in 32-bits, it will always be initialized
by default as a 32-bit integer.

.. code-block:: python

    >>> nd.ndarray(0)
    nd.ndarray(0, int32)

    >>> nd.ndarray(3.14)
    nd.ndarray(3.14, float64)

    >>> nd.ndarray(1 + 1j)
    nd.ndarray((1,1), complex<float64>)

Strings default to `blockref` strings, which are variable-sized strings.
The support for them is still preliminary, but some basic functionality
like converting between different unicode encodings is implemented.

One thing to note is that strings and unicode scalars are imported as
a view into the Python object's data. Because the Python object is immutable,
the ndarray also is flagged as immutable.

.. code-block:: python

    >>> nd.ndarray('testing')
    nd.ndarray("testing", string<ascii>)

    >>> nd.ndarray(u'testing')
    nd.ndarray("testing", string<ucs_2>)


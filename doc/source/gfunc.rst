DyND GFuncs
===========

NOTE: GFuncs are not currently operational due to refactoring
of the core object representation.

GFuncs are the array-programming primitive functions created and
used by blaze-local. They are currently in a very preliminary form,
but as they are they provide a preview of what is to come.

Lazy Evaluation
---------------

GFuncs and many other operations, like simple arithmetic, are
evaluated in a lazy function. This means that an expression
tree is built up, and the system has an opportunity to analyze
the complete expression DAG before evaluating the final values
when they are requested.

A simple illustration of this is to create an expression,
then evaluate it before and after modifying one of its inputs.

.. code-block:: python

    >>> a = nd.array([1,2])
    >>> b = a + 3
    >>> b
    nd.array([4, 5], int32)

    >>> a[0].val_assign(10)
    >>> b
    nd.array([13, 5], int32)

The evaluation engine is currently very primitive, and does not
support evaluating expressions with complicated expression graphs.
To evaluate such expressions, use the ``vals()`` method to create
intermediate results.

.. code-block:: python

    >>> (a + 3) * 2
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "_pydnd.pyx", line 289, in _pydnd.w_ndarray.__repr__ (D:\Develop\blaze\build\_pydnd.cxx:4170)
    RuntimeError: evaluating this expression graph is not yet supported:
    ("elementwise_binary_kernel",
      [SNIP]
    )

    >>> (a + 3).vals() * 2
    nd.array([8, 10], int32)

Using the Builtin GFuncs
------------------------

Blaze-local has a few gfuncs built in, at this stage primarily to
demonstrate the system. There is no implicit type promotion, so
any calls to the functions much have dtypes precisely matching
a function signature within the gfunc.

.. code-block:: python

    >>> nd.sin(3.1)
    nd.array(0.0415807, float64)

    >>> nd.sin(3)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "_pydnd.pyx", line 374, in _pydnd.w_elwise_gfunc.__call__ (D:\Develop\blaze\build\_pydnd.cxx:5735)
    RuntimeError: sin: could not find a gfunc kernel matching input argument types (int32)

    >>> nd.sum([1,3,5,9])
    nd.array(18, int32)

Creating Elementwise GFuncs
---------------------------

GFuncs are created using the constructors in the gfunc namespace. Once created, kernel
functions can be added, which presently must be provided as ctypes function pointer.
The ``nd.elwise_kernels`` namespace has a number of elementwise kernels to play with.

To demonstrate gfunc creation, we make a gfunc with a few different kernels that behave
differently. Normally one would want there to be a system to how overloads are done,
just as in C++, Java, or C#, this is only for demonstration purposes.

.. code-block:: python

    >>> myfunc = nd.gfunc.elwise('myfunc')
    >>> myfunc.add_kernel(nd.elwise_kernels.add_int32)
    >>> myfunc.add_kernel(nd.elwise_kernels.multiply_float64)
    >>> myfunc.add_kernel(nd.elwise_kernels.square_float64)

Now we can call the function if we provide operands of the right type.

.. code-block:: python

    >>> myfunc(3.0)
    nd.array(9, float64)

    >>> myfunc(1,2)
    nd.array(3, int32)

    >>> myfunc(1.0, 2.0)
    nd.array(2, float64)

    >>> myfunc(1, 2.0)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "_pydnd.pyx", line 374, in _pydnd.w_elwise_gfunc.__call__ (D:\Develop\blaze\build\_pydnd.cxx:5735)
    RuntimeError: myfunc: could not find a gfunc kernel matching input argument types (int32, float64)

Creating Elementwise Reduction GFuncs
-------------------------------------

Blaze-local supports simple reductions, like ``sum`` and ``min`` as well, through
elementwise reduction gfuncs. Presently, these require binary kernels whose output
is the same type as the two inputs. This will expand to other types in the future,
for example if an identity is provided, the output type could be different from the
input array.

We can make an example reduction that does ``sum`` or ``product`` depending on
the input dtype.

.. code-block:: python

    >>> myred = nd.gfunc.elwise_reduce('myred')
    >>> myred.add_kernel(nd.elwise_kernels.add_int32, associative=True, commutative=True, identity=0)
    >>> myred.add_kernel(nd.elwise_kernels.multiply_float64, associative=True, commutative=True, identity=1)

    >>> myred([1,2,3,4])
    nd.array(10, int32)
    >>> myred([1.,2.,3.,4.])
    nd.array(24, float64)

Groupby Reductions
------------------

Blaze-local has a simple ``nd.groupby`` function which, when combined with elementwise
reductions, can be used for groupby reductions. Here's a simple example.

.. code-block:: python

    >>> data = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    >>> by = np.array(['a', 'a', 'c', 'a', 'b', 'c', 'a', 'd'])
    >>> groups = nd.factor_categorical_dtype(by)
    >>> gb = nd.groupby(data, by, groups)

    >>> print(groups)
    categorical<fixedstring<ascii,1>, ["a", "b", "c", "d"]>

    >>> print("max:     ", nd.max(gb, axis=1))
    ('max:     ', nd.array([6, 4, 5, 7], int32))

    >>> print("min:     ", nd.min(gb, axis=1))
    ('min:     ', nd.array([0, 4, 2, 7], int32))

    >>> print("sum:     ", nd.sum(gb, axis=1))
    ('sum:     ', nd.array([10, 4, 7, 7], int32))

    >>> print("product: ", nd.product(gb, axis=1))
    ('product: ', nd.array([0, 4, 10, 7], int32))


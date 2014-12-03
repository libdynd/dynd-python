========================================
Converting Python Objects To DyND Arrays
========================================

In order to provide a user friendly boundary between Python objects and
DyND arrays, the Python exposure of LibDyND does some fairly extensive
analysis of input objects passed to the ``nd.array`` constructor. This
document provides an overview of the different ways this occurs.

There are a few ways provided to construct arrays from Python
objects, generally through the ``nd.array`` constructor. There
is also the ``nd.asarray`` function which is not as fully fleshed
out, which retains a view where possible such as when a buffer
interface is present.::

    >>> # Array from an object - fully autodetected
    >>> nd.array([1, 2, 3])
    nd.array([1, 2, 3], type="strided * int32")

    >>> # Array from an object - dtype provided
    >>> nd.array([1, 2, 3], dtype=ndt.int16)
    nd.array([1, 2, 3], type="strided * int16")

    >>> # Array from an object - full type provided
    >>> nd.array([1, 2, 3], type='strided * int16')
    nd.array([1, 2, 3], type="strided * int16")

The entry point for this is in the array class's ``__cinit__`` function
in Cython.

https://github.com/libdynd/dynd-python/blob/master/src/_pydynd.pyx#L971

This calls the ``array_init_from_pyobject`` function, which has two overloads
defined depending on whether a type is provided or not.

https://github.com/libdynd/dynd-python/blob/master/src/array_functions.cpp#L234

These call ``array_from_py``, similarly with and without type information::

https://github.com/libdynd/dynd-python/blob/master/src/array_from_py.cpp#L368

https://github.com/libdynd/dynd-python/blob/master/src/array_from_py.cpp#L592

Constructing With A Known Full Type
===================================

The simplest case is when the type of the desired array is fully known,
for example::

    >>> nd.array([1, 2, 3], type="strided * int16")
    nd.array([1, 2, 3], type="strided * int16")

To be able to do this conversion, DyND needs to do these steps:

* Traverse the object to determine its shape (3,)
* Create an empty ``nd.array`` using the shape information and type
* Copy the Python object into the array

This code is here:

https://github.com/libdynd/dynd-python/blob/master/src/array_from_py.cpp#L685

where it first checks whether the shape is required, and deduces
the shape, then calls ``array_nodim_broadcast_assign_from_py`` which
does an assignment from the pyobject without allowing new dimensions
to be broadcast. This function is defined in this file:

https://github.com/libdynd/dynd-python/blob/master/src/array_assign_from_py.cpp

Constructing with a Known DType
===============================

A little bit more complicated is the case to construct the array
when the dtype is known, but dimensions should be deduced from the
PyObject::

    >>> nd.array([1, 2, 3], dtype=ndt.int16)
    nd.array([1, 2, 3], type="strided * int16")

    >>> nd.array([['test', 1], ['two', 3]],
                 dtype='{name: string, value: int32}')
    nd.array([["test", 1], ["two", 3]], type="strided * {name : string, value : int32}")

To be able to do this, DyND needs to:

* Deduce the number of dimensions and shape of the object,
  using the dtype as a hint. This can be tricky, especially
  allowing for structure initialization from sequences.
* Create an empty ``nd.array`` using the shape information and type
* Copy the Python object into the array

This code is here:

https://github.com/libdynd/dynd-python/blob/master/src/array_from_py.cpp#L595

The code which deduces the shape and number of dimensions, then reconciles
it with the provided dtype is here:

https://github.com/libdynd/dynd-python/blob/master/src/array_from_py.cpp#L626

And finally the object is copied with the same function as in the
full type case, ``array_nodim_broadcast_assign_from_py``.

Constructing Fully Automatically
================================

When the PyObject isn't an instantly recognizable type, it gets to here:

https://github.com/libdynd/dynd-python/blob/master/src/array_from_py.cpp#L536

where it calls ``array_from_py_dynamic``. This function dynamically
updates the type, promoting both the dtype and dimension types
as it goes. This function is here:

https://github.com/libdynd/dynd-python/blob/master/src/array_from_py_dynamic.cpp

This function does need some more work, it does not support numpy
scalar types and arrays intermixed with iterators, for example.


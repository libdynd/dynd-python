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

https://github.com/ContinuumIO/dynd-python/blob/master/src/_pydynd.pyx#L971

This calls the ``array_init_from_pyobject`` function, which has two overloads
defined depending on whether a type is provided or not.

https://github.com/ContinuumIO/dynd-python/blob/master/src/array_functions.cpp#L234

These call ``array_from_py``, similarly with and without type information::

https://github.com/ContinuumIO/dynd-python/blob/master/src/array_from_py.cpp#L368

https://github.com/ContinuumIO/dynd-python/blob/master/src/array_from_py.cpp#L592

Constructing With A Known Type
==============================

The simplest case is when the type of the desired array is fully known,
for example::

    >>> nd.array([1, 2, 3], type="strided * int16")
    nd.array([1, 2, 3], type="strided * int16")

To be able to do this conversion, DyND needs to do these steps:

* Traverse the object to determine its shape (3,)
* Create an empty ``nd.array`` using the shape information and type
* Copy the Python object into the array



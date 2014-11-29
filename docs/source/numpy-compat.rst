===================
NumPy Compatibility
===================

While DyND doesn't aim to be a drop-in replacement for NumPy, having
a significant amount of compatibility is useful to help project
experiment with it. This document aims to track compatibility choices
that are made, both to be compatible and to have a different interface.

Type System
-----------

In NumPy, dtype and multi-dimensional array shape are separate properties
of the array.  DyND uses a different type system, developed within the
Blaze project and called datashape. Dimensions are a part of the types,
and the shape is typically encoded in the type as well.

Basic Constructors
------------------

The constructors ``nd.empty``, ``nd.zeros``, and ``nd.ones``, are very
similar to the NumPy functions of the same name. There is compatibility
in the sense that passing a shape as a tuple followed by a dtype works
in both. DyND is more flexible about the input parameters though.

The DyND constructors require a dtype be specified, they do not default
to float64 as in NumPy.

The ``nd.full`` constructor has an incompatible signature with NumPy.

Array Attributes
----------------

DyND arrays expose ``ndim``, ``shape``, ``strides``, and ``dtype``.

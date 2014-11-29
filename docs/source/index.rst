.. DyND-Python documentation master file, created by
   sphinx-quickstart on Fri Aug 17 17:12:56 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the DyND Python bindings documentation!
===================================

.. warning::

   DyND is in a preview release stage, things are expected
   to change significantly during pre-1.0 development.

This is some preliminary documentation for DyND in Python, meant to provide
a simple reference to some of the key features in the library. Not everything
in the library is covered, and many things you might expect to work are
not implemented yet.

In any examples, it is assumed that ``from dynd import nd, ndt`` has been
executed prior to the sample Python code. In examples demonstrating
Numpy interoperability, ``import numpy as np`` is also assumed.

Contents:

.. toctree::
    :maxdepth: 2

    type
    array
    gfunc
    numpy-compat
    debugging
    dev
    dev-frompython

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


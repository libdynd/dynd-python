================================
DyND Python Bindings Development
================================

The DyND library, in its initial conception, is trying to
work well in two worlds, as both a C++ library and
a Python module. In both languages, DyND should feel
like a native first-class project, interacting with
the features of each language in the way heavy users
of the language will expect intuitively.
CPython is the present target for the Python bindings.

In Python, this means that the library should
exercise fine-grained control over how it interacts
with all the native Python types and interfaces.
NumPy accomplishes this by working with the CPython
API directly. For DyND, we did not want to go this
route because it requires a large amount of boilerplate
code.

Cython + CPython API for Bindings
---------------------------------

The approach chosen for DyND is to use Cython for
its ability to generate the CPython API boilerplate
for extension classes and other constructs, combined
with directly using the CPython API from C++ for
the rest. A minimal set of support classes and functions,
such as an RAII PyObject* class, are used to reduce
potential errors in accessing the API.

A number of techniques are used to make things integrate
smoothly and perform well. Let's go through them one
by one.

See http://docs.cython.org/src/userguide/wrapping_CPlusPlus.html
for the Cython documentation about wrapping C++.

Only Thin Wrappers in Cython
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This was a choice made after trying to do a few things in
Cython, and running into troubles with Cython's support
of C++ features like pointers, references, and operator
overloading. The Cython .pyx file mostly consists of
class definitions, docstrings, and calls to C++ functions.

Inline Functions
~~~~~~~~~~~~~~~~

For functionality we would like to inline into
the Cython functions, but would like to write in
C++ because Cython doesn't support the appropriate
syntax, or C++ is easier, we can use inline functions.

An example where this is necessary is to do
an assignment statement. The Cython developers debated
where to draw the line on this feature, whether
Cython should be more Python-like or C++-like, and
the result is that some code must be in C++ to access
C++ features.

Placement New/Delete
~~~~~~~~~~~~~~~~~~~~

The way the Cython documentation recommends that C++
objects be wrapped is to allocate them on the heap
and store a pointer in the Python object.

http://docs.cython.org/src/userguide/wrapping_CPlusPlus.html#create-cython-wrapper-class

Most objects in DyND behave in a smart pointer fashion,
so storing them with a heap allocation would add an
extra inefficient memory allocation and pointer indirection.

Cython does not allow C++ objects to be placed directly
in its data, only POD C structs. We can still accomplish
what we desire by manually calling the placement new
and destructor. In `placement_wrappers.hpp`, a set
of inline functions is defined to do placement new,
delete, get a reference to the C++ value, and
assign the C++ value.

This slightly obfuscates the Cython code, but the
names have been chosen in a way which is working
well in practice. Here's how the dtype class wrapper
starts::

    cdef class w_type:
        cdef ndt_type_placement_wrapper v

        def __cinit__(self, rep=None):
            placement_new(self.v)
            if rep is not None:
                SET(self.v, make_dtype_from_pyobject(rep))
        def __dealloc__(self):
            placement_delete(self.v)

Accessing Type Objects from C++
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For proper integration, its necessary to have access to
each wrapper class's type object, with corresponding
`<classname>_Check` function.

Cython has a mechanism to automatically generate a
header file, but in a test of generating a header for
just one of the wrapped classes, it generated a 1300 line
header, and still required calling a special `import_dynd`
function from the C++ side. This mechanism is designed for
allowing third party libraries to access the classes, but even
for that purpose a handwritten header seems much more appropriate,
as it would be human readable.

There is just one piece of information the C++ code needs to learn
from Cython, the TypeObject instance. Ideally one could declare an
exported variable `PyTypeObject *WArray_Type` in the Cython code,
and reference it with a one-liner in a header file. The Cython syntax
doesn't appear to support this, so the alternative being used is to
declare a C++ function::

    PyTypeObject *pydynd::WArray_Type;

    void pydynd::init_w_array_typeobject(PyObject *type)
    {
        WArray_Type = (PyTypeObject *)type;
    }

and call it from Cython at the outer scope, which is executed
on initialization::

    init_w_array_typeobject(w_array)
    
The full implementation of this in the C++ header is then::

    extern PyTypeObject *WArray_Type;
    inline bool WArray_CheckExact(PyObject *obj) {
        return Py_TYPE(obj) == WArray_Type;
    }
    inline bool WArray_Check(PyObject *obj) {
        return PyObject_TypeCheck(obj, WArray_Type);
    }
    struct WArray {
      PyObject_HEAD;
      // This is array_placement_wrapper in Cython-land
      dynd::array v;
    };
    void init_w_array_typeobject(PyObject *type);

There is a special consideration that must be made when constructing
the Cython classes from C++, which is that calling the
`WArray_Type->tp_alloc` method does not call the Cython
`__cinit__` function. This leads to the following wrapper code::

    inline PyObject *wrap_array(const dynd::array& n) {
        WArray *result = (WArray *)WArray_Type->tp_alloc(WArray_Type, 0);
        if (!result) {
            throw std::runtime_error("");
        }
        // Calling tp_alloc doesn't call Cython's __cinit__, so do the placement new here
        pydynd::placement_new(reinterpret_cast<pydynd::array_placement_wrapper &>(result->v));
        result->v = n;
        return (PyObject *)result;
    }

Translating C++ Exceptions to Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

http://docs.cython.org/src/userguide/wrapping_CPlusPlus.html#exceptions

Cython supports an automatic mechanism for translating
C++ exceptions into Python exceptions. The default
way to handle this translation is when declaring
functions imported from header files, to add 'except +'
to the end of the definition, as follows::

    void pydynd::translate_exception()
    {
        try {
            if (PyErr_Occurred())
                ; // let the latest Python exn pass through and ignore the current one
            else
                throw;
        } catch (const dynd::broadcast_error& exn) {
            PyErr_SetString(BroadcastException, exn.message());
        } catch (const dynd::too_many_indices& exn) {
            PyErr_SetString(PyExc_IndexError, exn.message());
        ...
        } catch (const std::exception& exn) {
            PyErr_SetString(PyExc_RuntimeError, exn.what());
        }
    }

The naked `throw` reraises the exception caught by the Cython code,
and uses an appropriate PyErr_SetString or PyErr_SetObject
to translate the exception. It appears that this is
conformant C++, as it is rethrowing the exception while within
scope of another catch statement, even though that statement
is within another function.

The only problem encountered is on Mac OS X, on an older version of
clang, where catching subclasses don't appear to work, and explicit
catches of every single possible exception was required. The solution
at the time was to switch to using g++ 4.2.

Defining Custom Python Exceptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The standard Python exceptions do not cover all the cases needed
by DyND, so we need to define some additional exception types
These new exceptions are defined in Cython, and their `TypeObject`
is passed to C++ in the same way as others. Here is the Cython
code for the `BroadcastError` class::

    # Exceptions to convert from C++
    class BroadcastError(Exception):
        pass

    # Register all the exception objects with the exception translator
    set_broadcast_exception(BroadcastError)

and the corresponding C++ code::

    PyObject *BroadcastException = NULL;

    void pydynd::set_broadcast_exception(PyObject *e)
    {
        BroadcastException = e;
    }

Accessing CTypes Structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~

CTypes doesn't define a C API for accessing its objects, so
to provide integration with CTypes requires some additional
work. This is done by defining a struct with all the needed
`PyTypeObject` instances from CTypes, and initializing them
using the CPython API at startup.::

    /**
     * Struct with data about the _ctypes module.
     */
    struct ctypes_info {
        // The _ctypes module (for C-implementation details)
        PyObject *_ctypes;
        // These match the corresponding names within _ctypes.c
        PyObject *PyCData_Type;
        PyObject *PyCStructType_Type;
        PyObject *UnionType_Type;
        PyObject *PyCPointerType_Type;
        PyObject *PyCArrayType_Type;
        PyObject *PyCSimpleType_Type;
        PyObject *PyCFuncPtrType_Type;
    };

    extern ctypes_info ctypes;

    /**
     * Should be called at module initialization, this
     * stores some internal information about the ctypes
     * classes for later.
     */
    void init_ctypes_interop();

PEP 3118 / Python Buffer Protocol
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For images and matrices, the buffer protocol introduced
in Python 2.6 is a great way to expose and consume regular
array data. This mechanism supports communicating multi-dimensional
arrays between C modules. For any module, like DyND, which exposes
images, matrices, or other multi-dimensional strided data,
supporting this is mandatory to interoperate properly with NumPy,
Cython, and other Python numerical libraries.

There are still some rough edges in the specification and
implementation. In the official Python 3.3 documentation, the
buffer protocol refers to the `struct` module for the specification
of the `format` string, but the `struct` module doesn't
include some of the additions proposed in PEP 3118, such as
complex numbers. Because most programs are still using this
protocol for low-level communication of arrays, supporting
16-bit floating point, complex, and other types specified
in PEP 3118 is possible without requiring the Python `struct`
module to support everything.


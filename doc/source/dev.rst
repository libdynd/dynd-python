================================
DyND Python Bindings Development
================================

The DyND library, in its initial conception, is trying to
work well in two worlds as both a C++ library and
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

    cdef class w_dtype:
        cdef dtype_placement_wrapper v

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
exported variable `PyTypeObject *WNDObject_Type` in the Cython code,
and reference it with a one-liner in a header file. The Cython syntax
doesn't appear to support this, so the alternative being used is to
declare a C++ function::

    PyTypeObject *pydynd::WNDObject_Type;

    void pydynd::init_w_ndobject_typeobject(PyObject *type)
    {
        WNDObject_Type = (PyTypeObject *)type;
    }

and call it from Cython at the outer scope, which is executed
on initialization::

    init_w_ndobject_typeobject(w_ndobject)
    
The full implementation of this in the C++ header is then::

    extern PyTypeObject *WNDObject_Type;
    inline bool WNDObject_CheckExact(PyObject *obj) {
        return Py_TYPE(obj) == WNDObject_Type;
    }
    inline bool WNDObject_Check(PyObject *obj) {
        return PyObject_TypeCheck(obj, WNDObject_Type);
    }
    struct WNDObject {
      PyObject_HEAD;
      // This is ndobject_placement_wrapper in Cython-land
      dynd::ndobject v;
    };
    void init_w_ndobject_typeobject(PyObject *type);

There is a special consideration that must be made when constructing
the Cython classes from C++, which is that calling the
`WNDObject_Type->tp_alloc` method does not call the Cython
`__cinit__` function. This leads to the following wrapper code::

    inline PyObject *wrap_ndobject(const dynd::ndobject& n) {
        WNDObject *result = (WNDObject *)WNDObject_Type->tp_alloc(WNDObject_Type, 0);
        if (!result) {
            throw std::runtime_error("");
        }
        // Calling tp_alloc doesn't call Cython's __cinit__, so do the placement new here
        pydynd::placement_new(reinterpret_cast<pydynd::ndobject_placement_wrapper &>(result->v));
        result->v = n;
        return (PyObject *)result;
    }

Only Thin Wrappers in Cython
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This was a choice made after trying to do a few things in
Cython, and running into troubles with Cython's support
of C++ features like pointers, references, and operator
overloading. In a nearly universal fashion, the Cython .pyx
file only contains the class definitions, docstrings,
and calls to C++ functions.

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
to translate the exception. I'm not sure whether this is
conformant C++, but it appears to work well on all the compilers
Cython is supporting.

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

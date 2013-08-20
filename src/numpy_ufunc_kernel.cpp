//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//


#include "numpy_ufunc_kernel.hpp"

#if defined(_MSC_VER)
#pragma warning(push,2)
#endif
#include <numpy/npy_3kcompat.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif


#include "utility_functions.hpp"

#if DYND_NUMPY_INTEROP

using namespace std;
using namespace dynd;
using namespace pydynd;

PyObject *pydynd::numpy_typetuples_from_ufunc(PyObject *ufunc)
{
    // NOTE: This function does not raise C++ exceptions,
    //       it behaves as a Python C-API function.
    if (!PyObject_TypeCheck(ufunc, &PyUFunc_Type)) {
        PyErr_SetString(PyExc_TypeError, "require a numpy ufunc object "
                        "to retrieve its type tuples");
        return NULL;
    }
    PyUFuncObject *uf = (PyUFuncObject *)ufunc;
    if (uf->core_enabled) {
        PyErr_SetString(PyExc_TypeError, "gufunc type tuple extraction is not supported yet");
        return NULL;
    }

    // Process the main ufunc loops list
    int builtin_count = uf->ntypes;
    int nargs = uf->nin + uf->nout;
    PyObject *result = PyList_New(builtin_count);
    if (result == NULL) {
        return NULL;
    }
    for (int i = 0; i < builtin_count; ++i) {
        PyObject *typetup = PyTuple_New(nargs);
        if (typetup == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        char *types = uf->types + i * nargs;
        for (int j = 0; j < nargs; ++j) {
            PyObject *descr = (PyObject *)PyArray_DescrFromType(types[j]);
            if (descr == NULL) {
                Py_DECREF(result);
                Py_DECREF(typetup);
                return NULL;
            }
            PyTuple_SET_ITEM(typetup, j, descr);
        }
        PyList_SET_ITEM(result, i, typetup);
    }

    // Process the userloops as well
    if (uf->userloops != NULL) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(uf->userloops, &pos, &key, &value)) {
            PyUFunc_Loop1d *funcdata = (PyUFunc_Loop1d *)NpyCapsule_AsVoidPtr(value);
            while (funcdata != NULL) {
                PyObject *typetup = PyTuple_New(nargs);
                if (typetup == NULL) {
                    Py_DECREF(result);
                    return NULL;
                }
                int *types = funcdata->arg_types;
                for (int j = 0; j < nargs; ++j) {
                    PyObject *descr = (PyObject *)PyArray_DescrFromType(types[j]);
                    if (descr == NULL) {
                        Py_DECREF(result);
                        Py_DECREF(typetup);
                        return NULL;
                    }
                    PyTuple_SET_ITEM(typetup, j, descr);
                }
                PyList_Append(result, typetup);

                funcdata = funcdata->next;
            }
        }
    }

    return result;
}


#endif // DYND_NUMPY_INTEROP
//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//


#include "numpy_ufunc_kernel.hpp"
#include <numpy/npy_3kcompat.h>

#include "utility_functions.hpp"

#if DYND_NUMPY_INTEROP

using namespace std;
using namespace dynd;
using namespace pydynd;

PyObject *pydynd::numpy_typetuples_from_ufunc(PyObject *ufunc)
{
    if (!PyObject_TypeCheck(ufunc, &PyUFunc_Type)) {
        PyErr_SetString(PyExc_TypeError, "require a numpy ufunc object "
                        "to retrieve its type tuples");
        return NULL;
    }
    PyUFuncObject *uf = (PyUFuncObject *)ufunc;
cout << "uf: " << (void *)uf << endl;
    if (uf->core_enabled) {
        PyErr_SetString(PyExc_TypeError, "gufunc type tuple extraction is not supported yet");
        return NULL;
    }
cout << "line: " << __LINE__ << endl;

    // Process the main ufunc loops list
    int builtin_count = uf->ntypes;
    int nargs = uf->nin + uf->nout;
    PyObject *result = PyList_New(builtin_count);
    if (result == NULL) {
        return NULL;
    }
    for (int i = 0; i < builtin_count; ++i) {
cout << "line: " << __LINE__ << endl;
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
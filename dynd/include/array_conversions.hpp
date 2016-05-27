//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

/* This file exists to expose whatever functionality from
 * nd/array.pyx and nd/callable.pyx needs to be available
 * in the C++ portions of the codebase.
 * Currently it exists only as a means of
 * exporting conversions between the Cython wrapper
 * types and their corresponding C++ types.
 * Those conversions must be exported from Cython
 * to the C++ code since the ABI for the wrapper types
 * must be defined there (and depends on Cython's ABI for
 * extension types).
 *
 * The conversions defined here lazily load the modules
 * that expose the core ABI for the wrapper types.
*/

#pragma once

#include <Python.h>

#include <dynd/array.hpp>
#include <dynd/callable.hpp>

#include "visibility.hpp"

namespace pydynd {

PYDYND_API dynd::nd::array &array_to_cpp_ref(PyObject *);
PYDYND_API PyTypeObject *get_array_pytypeobject();
PYDYND_API PyObject *array_from_cpp(const dynd::nd::array &);

PYDYND_API dynd::nd::callable &callable_to_cpp_ref(PyObject *);
PYDYND_API PyTypeObject *get_callable_pytypeobject();
PYDYND_API PyObject *callable_from_cpp(const dynd::nd::callable &);

} // namespace pydynd

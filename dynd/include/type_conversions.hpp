//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

/* This file exists to expose whatever functionality from
 * ndt/type.pyx needs to be available throughout the
 * C++ codebase. Currently it exists only as a means
 * of providing conversions between the Cython wrapper
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

#include <dynd/type.hpp>

#include "visibility.hpp"

namespace pydynd {

PYDYND_API dynd::ndt::type &type_to_cpp_ref(PyObject *);
PYDYND_API PyTypeObject *get_type_pytypeobject();
PYDYND_API PyObject *type_from_cpp(const dynd::ndt::type &);
PYDYND_API dynd::ndt::type dynd_ndt_as_cpp_type(PyObject *);
PYDYND_API dynd::ndt::type dynd_ndt_cpp_type_for(PyObject *);

} // namespace pydynd

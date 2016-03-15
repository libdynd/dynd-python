//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
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
PYDYND_API dynd::ndt::type ndt_type_from_pylist(PyObject *);

} // namespace pydynd

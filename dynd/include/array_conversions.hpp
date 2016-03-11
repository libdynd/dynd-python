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

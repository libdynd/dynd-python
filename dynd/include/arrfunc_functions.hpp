//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// This header defines some wrapping functions to
// access various nd::callable parameters
//

#pragma once

#include <Python.h>

#include <sstream>

#include <dynd/func/callable.hpp>

#include "config.hpp"
#include "array_from_py.hpp"
#include "array_as_py.hpp"
#include "array_as_numpy.hpp"
#include "array_as_pep3118.hpp"
#include "eval_context_functions.hpp"
#include "array_functions.hpp"

#include "visibility.hpp"
#include "wrapper.hpp"

typedef DyND_PyWrapperObject<dynd::nd::callable> DyND_PyCallableObject;

inline int DyND_PyCallable_Check(PyObject *obj)
{
  return DyND_PyWrapper_Check<dynd::nd::callable>(obj);
}

inline int DyND_PyCallable_CheckExact(PyObject *obj)
{
  return DyND_PyWrapper_CheckExact<dynd::nd::callable>(obj);
}

namespace pydynd {

PYDYND_API void init_w_callable_typeobject(PyObject *type);

PYDYND_API PyObject *callable_call(PyObject *af_obj, PyObject *args_obj,
                                   PyObject *kwds_obj, PyObject *ectx_obj);

PyObject *callable_rolling_apply(PyObject *func_obj, PyObject *arr_obj,
                                 PyObject *window_size_obj, PyObject *ectx_obj);

/**
 * Returns a dictionary of all the published callables.
 */
PYDYND_API PyObject *get_published_callables();

PYDYND_API PyObject *callable_str(const dynd::nd::callable &f);

} // namespace pydynd

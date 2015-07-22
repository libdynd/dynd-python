//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// This header defines some wrapping functions to
// access various nd::arrfunc parameters
//

#pragma once

#include <Python.h>

#include <sstream>

#include <dynd/func/arrfunc.hpp>

#include "config.hpp"
#include "array_from_py.hpp"
#include "array_as_py.hpp"
#include "array_as_numpy.hpp"
#include "array_as_pep3118.hpp"
#include "eval_context_functions.hpp"
#include "array_functions.hpp"

namespace pydynd {

/**
 * This is the typeobject and struct of w_arrfunc from Cython.
 */
extern PyTypeObject *WArrFunc_Type;
inline bool WArrFunc_CheckExact(PyObject *obj)
{
  return Py_TYPE(obj) == WArrFunc_Type;
}
inline bool WArrFunc_Check(PyObject *obj)
{
  return PyObject_TypeCheck(obj, WArrFunc_Type);
}

struct WArrFunc {
  PyObject_HEAD;

  // This is array_placement_wrapper in Cython-land
  dynd::nd::arrfunc v;
};

PYDYND_API inline PyObject *wrap_arrfunc(const dynd::nd::arrfunc &n)
{
  WArrFunc *result = (WArrFunc *) WArrFunc_Type->tp_alloc(WArrFunc_Type, 0);
  if (!result) {
    throw std::runtime_error("");
  }

  result->v = n;
  return reinterpret_cast<PyObject *>(result);
}


PYDYND_API void init_w_arrfunc_typeobject(PyObject *type);

PYDYND_API PyObject *arrfunc_call(PyObject *af_obj, PyObject *args_obj,
                                  PyObject *kwds_obj, PyObject *ectx_obj);

PyObject *arrfunc_rolling_apply(PyObject *func_obj, PyObject *arr_obj,
                                PyObject *window_size_obj, PyObject *ectx_obj);

/**
 * Returns a dictionary of all the published arrfuncs.
 */
PyObject *get_published_arrfuncs();

} // namespace pydynd
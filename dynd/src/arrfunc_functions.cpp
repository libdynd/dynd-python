//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "arrfunc_functions.hpp"
#include "array_functions.hpp"
#include "array_from_py.hpp"
#include "array_assign_from_py.hpp"
#include "type_functions.hpp"
#include "utility_functions.hpp"
#include "numpy_interop.hpp"
#include "arrfunc_from_pyfunc.hpp"

#include <dynd/types/string_type.hpp>
#include <dynd/types/base_dim_type.hpp>
#include <dynd/memblock/external_memory_block.hpp>
#include <dynd/array_range.hpp>
#include <dynd/type_promotion.hpp>
#include <dynd/types/base_struct_type.hpp>
#include <dynd/types/base_bytes_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/func/rolling.hpp>
#include <dynd/view.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/func/callable_registry.hpp>

using namespace std;
using namespace dynd;
using namespace pydynd;

void pydynd::init_w_callable_typeobject(PyObject *type)
{
  DyND_PyWrapper_Type<dynd::nd::callable>() = (PyTypeObject *)type;
}

PyObject *pydynd::callable_call(PyObject *af_obj, PyObject *args_obj,
                                PyObject *kwds_obj, PyObject *ectx_obj)
{
  if (!DyND_PyCallable_Check(af_obj)) {
    PyErr_SetString(PyExc_TypeError, "callable_call expected an nd.callable");
    return NULL;
  }
  dynd::nd::callable &af = ((DyND_PyCallableObject *)af_obj)->v;
  if (af.is_null()) {
    PyErr_SetString(PyExc_ValueError, "cannot call a null nd.callable");
    return NULL;
  }
  if (!PyTuple_Check(args_obj)) {
    PyErr_SetString(PyExc_ValueError,
                    "callable_call requires a tuple of arguments");
    return NULL;
  }
  if (!PyDict_Check(kwds_obj)) {
    PyErr_SetString(PyExc_ValueError,
                    "callable_call requires a dictionary of keyword arguments");
    return NULL;
  }
  const eval::eval_context *ectx = eval_context_from_pyobj(ectx_obj);

  // Convert args into nd::arrays
  intptr_t narg = PyTuple_Size(args_obj);
  std::vector<dynd::nd::array> arg_values(narg);
  for (intptr_t i = 0; i < narg; ++i) {
    arg_values[i] =
        array_from_py(PyTuple_GET_ITEM(args_obj, i), 0, false, ectx);
  }

  // Convert kwds into nd::arrays
  intptr_t nkwd = PyDict_Size(kwds_obj);
  vector<std::string> kwd_names_strings(nkwd);
  std::vector<const char *> kwd_names(nkwd);
  std::vector<dynd::nd::array> kwd_values(nkwd);
  PyObject *key, *value;
  for (Py_ssize_t i = 0, j = 0; PyDict_Next(kwds_obj, &i, &key, &value); ++j) {
    kwd_names_strings[j] = pystring_as_string(key);
    kwd_names[j] = kwd_names_strings[j].c_str();
    kwd_values[j] = array_from_py(value, 0, false, ectx);
  }

  dynd::nd::array result =
      af(narg, arg_values.empty() ? NULL : arg_values.data(),
         kwds(nkwd, kwd_names.empty() ? NULL : kwd_names.data(),
              kwd_values.empty() ? NULL : kwd_values.data()));
  return DyND_PyWrapper_New(result);
}

PyObject *pydynd::callable_rolling_apply(PyObject *func_obj, PyObject *arr_obj,
                                         PyObject *window_size_obj,
                                         PyObject *ectx_obj)
{
  eval::eval_context *ectx =
      const_cast<eval::eval_context *>(eval_context_from_pyobj(ectx_obj));
  dynd::nd::array arr = array_from_py(arr_obj, 0, false, ectx);
  intptr_t window_size = pyobject_as_index(window_size_obj);
  dynd::nd::callable func;
  if (DyND_PyCallable_Check(func_obj)) {
    func = ((DyND_PyCallableObject *)func_obj)->v;
  } else {
    ndt::type el_tp = arr.get_type().get_type_at_dimension(NULL, 1);
    ndt::type proto = ndt::callable_type::make(
        el_tp, ndt::tuple_type::make(ndt::make_fixed_dim_kind(el_tp)));

    func = pydynd::nd::functional::apply(func_obj, proto);
  }
  dynd::nd::callable roll = dynd::nd::functional::rolling(func, window_size);
  dynd::nd::array result = roll(arr);
  return DyND_PyWrapper_New(result);
}

PyObject *pydynd::get_published_callables()
{
  pyobject_ownref res(PyDict_New());
  const map<std::string, dynd::nd::callable> &reg = func::get_regfunctions();
  for (map<std::string, dynd::nd::callable>::const_iterator it = reg.begin();
       it != reg.end(); ++it) {
    PyDict_SetItem(res.get(), pystring_from_string(it->first),
                   DyND_PyWrapper_New(it->second));
  }
  return res.release();
}

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
#include <dynd/types/struct_type.hpp>
#include <dynd/types/base_bytes_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/func/rolling.hpp>
#include <dynd/view.hpp>
#include <dynd/callable.hpp>
#include <dynd/func/callable_registry.hpp>

using namespace std;
using namespace dynd;
using namespace pydynd;

PyObject *pydynd::callable_call(PyObject *af_obj, PyObject *args_obj,
                                PyObject *kwds_obj)
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

  // Convert args into nd::arrays
  intptr_t narg = PyTuple_Size(args_obj);
  std::vector<dynd::nd::array> arg_values(narg);
  for (intptr_t i = 0; i < narg; ++i) {
    arg_values[i] = array_from_py(PyTuple_GET_ITEM(args_obj, i), 0, false);
  }

  // Convert kwds into nd::arrays
  intptr_t nkwd = PyDict_Size(kwds_obj);
  vector<std::string> kwd_names_strings(nkwd);
  std::vector<std::pair<const char *, dynd::nd::array>> kwds2(nkwd);
  PyObject *key, *value;
  for (Py_ssize_t i = 0, j = 0; PyDict_Next(kwds_obj, &i, &key, &value); ++j) {
    kwd_names_strings[j] = pystring_as_string(key);
    kwds2[j].first = kwd_names_strings[j].c_str();
    kwds2[j].second = array_from_py(value, 0, false);
  }

  dynd::nd::array result =
      af.call(narg, arg_values.empty() ? NULL : arg_values.data(), nkwd,
              kwds2.empty() ? NULL : kwds2.data());
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

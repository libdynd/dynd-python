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
#include <dynd/func/rolling_arrfunc.hpp>
#include <dynd/view.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/func/arrfunc_registry.hpp>

using namespace std;
using namespace dynd;
using namespace pydynd;

PyTypeObject *pydynd::WArrFunc_Type;

void pydynd::init_w_arrfunc_typeobject(PyObject *type)
{
  WArrFunc_Type = (PyTypeObject *)type;
}

PyObject *pydynd::arrfunc_call(PyObject *af_obj, PyObject *args_obj,
                               PyObject *ectx_obj)
{
  if (!WArrFunc_Check(af_obj)) {
    PyErr_SetString(PyExc_TypeError, "arrfunc_call expected an nd.arrfunc");
    return NULL;
  }
  const nd::arrfunc &af = ((WArrFunc *)af_obj)->v;
  if (af.is_null()) {
    PyErr_SetString(PyExc_ValueError, "cannot call a null nd.arrfunc");
    return NULL;
  }
  if (!PyTuple_Check(args_obj)) {
    PyErr_SetString(PyExc_ValueError,
                    "arrfunc_call requires a tuple of arguments");
    return NULL;
  }
  const eval::eval_context *ectx = eval_context_from_pyobj(ectx_obj);
  // Convert args into nd::arrays
  intptr_t args_size = PyTuple_Size(args_obj);
  std::vector<nd::array> args(args_size);
  for (intptr_t i = 0; i < args_size; ++i) {
    args[i] = array_from_py(PyTuple_GET_ITEM(args_obj, i), 0, false, ectx);
  }

  // add in ectx
  nd::array result = af(static_cast<intptr_t>(args_size), static_cast<nd::array *>(args_size ? &args[0] : NULL));
  return wrap_array(result);
}

PyObject *pydynd::arrfunc_rolling_apply(PyObject *func_obj, PyObject *arr_obj,
                                        PyObject *window_size_obj,
                                        PyObject *ectx_obj)
{
  eval::eval_context *ectx = const_cast<eval::eval_context *>(eval_context_from_pyobj(ectx_obj));
  nd::array arr = array_from_py(arr_obj, 0, false, ectx);
  intptr_t window_size = pyobject_as_index(window_size_obj);
  nd::arrfunc func;
  if (WArrFunc_Check(func_obj)) {
    func = ((WArrFunc *)func_obj)->v;
  }
  else {
    ndt::type el_tp = arr.get_type().get_type_at_dimension(NULL, 1);
    ndt::type proto = ndt::make_arrfunc(
        ndt::make_tuple(ndt::make_fixed_dimsym(el_tp)), el_tp);

    func = arrfunc_from_pyfunc(func_obj, proto);
  }
  nd::arrfunc roll = make_rolling_arrfunc(func, window_size);
  nd::array result = roll(arr, kwds("ectx", ectx));
  return wrap_array(result);
}

PyObject *pydynd::get_published_arrfuncs()
{
  pyobject_ownref res(PyDict_New());
  const map<nd::string, nd::arrfunc> &reg = func::get_regfunctions();
  for (map<nd::string, nd::arrfunc>::const_iterator it = reg.begin();
       it != reg.end(); ++it) {
    PyDict_SetItem(res.get(), pystring_from_string(it->first.str()),
                   wrap_array(it->second));
  }
  return res.release();
}

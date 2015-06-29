//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "eval_context_functions.hpp"
#include "utility_functions.hpp"

using namespace std;
using namespace dynd;
using namespace pydynd;

PyTypeObject *pydynd::WEvalContext_Type;

void pydynd::init_w_eval_context_typeobject(PyObject *type)
{
  WEvalContext_Type = (PyTypeObject *)type;
}

static void modify_eval_context(eval::eval_context *ectx, PyObject *kwargs)
{
  if (!PyDict_Check(kwargs)) {
    throw invalid_argument(
        "nd.eval_context(): invalid kwargs, expected a dict");
  }
  if (PyDict_Size(kwargs) == 0) {
    return;
  }
  PyObject *obj;
  // reset to factory settings
  obj = PyDict_GetItemString(kwargs, "reset");
  if (obj != NULL) {
    if (PyObject_IsTrue(obj)) {
      // Reset to factory settings
      *ectx = eval::eval_context();
    }
    if (PyDict_DelItemString(kwargs, "reset") < 0) {
      throw runtime_error("");
    }
  }
  // errmode
  obj = PyDict_GetItemString(kwargs, "errmode");
  if (obj != NULL) {
    ectx->errmode = pyarg_error_mode_no_default(obj);
    if (PyDict_DelItemString(kwargs, "errmode") < 0) {
      throw runtime_error("");
    }
  }
  // cuda_device_errmode
  obj = PyDict_GetItemString(kwargs, "cuda_device_errmode");
  if (obj != NULL) {
    ectx->cuda_device_errmode = pyarg_error_mode_no_default(obj);
    if (PyDict_DelItemString(kwargs, "cuda_device_errmode") < 0) {
      throw runtime_error("");
    }
  }
  // date_parse_order
  obj = PyDict_GetItemString(kwargs, "date_parse_order");
  if (obj != NULL) {
    ectx->date_parse_order = (date_parse_order_t)pyarg_strings_to_int(
        obj, "date_parse_order", date_parse_no_ambig, "NoAmbig",
        date_parse_no_ambig, "YMD", date_parse_ymd, "MDY", date_parse_mdy,
        "DMY", date_parse_dmy);
    if (PyDict_DelItemString(kwargs, "date_parse_order") < 0) {
      throw runtime_error("");
    }
  }
  // century_window
  obj = PyDict_GetItemString(kwargs, "century_window");
  if (obj != NULL) {
    long century_window = PyLong_AsLong(obj);
    if (century_window < 0 || (century_window > 99 && century_window < 1000)) {
      stringstream ss;
      ss << "nd.eval_context(): invalid century_window value ";
      ss << century_window << ", must be 0 (no two digit year handling)";
      ss << ", 1-99 (sliding window), or 1000 and up (fixed window)";
      throw invalid_argument(ss.str());
    }
    ectx->century_window = (int)century_window;
    if (PyDict_DelItemString(kwargs, "century_window") < 0) {
      throw runtime_error("");
    }
  }

  // Verify that there are no more keyword arguments
  PyObject *key, *value;
  Py_ssize_t pos = 0;

  while (PyDict_Next(kwargs, &pos, &key, &value)) {
    stringstream ss;
    ss << "nd.eval_context(): got an unexpected keyword argument ";
    ss << "'" << pystring_as_string(key) << "'";
    throw invalid_argument(ss.str());
  }
}

eval::eval_context *pydynd::new_eval_context(PyObject *kwargs)
{
  // Allocate the eval_context, copying eval::default_eval_context to start
  eval::eval_context ectx(eval::default_eval_context);

  // Validate the kwargs is a non-empty dictionary
  if (kwargs == NULL || kwargs == Py_None) {
    return new eval::eval_context(ectx);
  }

  modify_eval_context(&ectx, kwargs);

  return new eval::eval_context(ectx);
}

void pydynd::modify_default_eval_context(PyObject *kwargs)
{
  modify_eval_context(&eval::default_eval_context, kwargs);
}

PyObject *pydynd::get_eval_context_errmode(PyObject *ectx_obj)
{
  if (!WEvalContext_Check(ectx_obj)) {
    throw invalid_argument("expected an nd.eval_context object");
  }
  const eval::eval_context *ectx = ((WEvalContext *)ectx_obj)->ectx;
  return pyarg_error_mode_to_pystring(ectx->errmode);
}

PyObject *pydynd::get_eval_context_cuda_device_errmode(PyObject *ectx_obj)
{
  if (!WEvalContext_Check(ectx_obj)) {
    throw invalid_argument("expected an nd.eval_context object");
  }
  const eval::eval_context *ectx = ((WEvalContext *)ectx_obj)->ectx;
  return pyarg_error_mode_to_pystring(ectx->cuda_device_errmode);
}

PyObject *pydynd::get_eval_context_date_parse_order(PyObject *ectx_obj)
{
  if (!WEvalContext_Check(ectx_obj)) {
    throw invalid_argument("expected an nd.eval_context object");
  }
  const eval::eval_context *ectx = ((WEvalContext *)ectx_obj)->ectx;
  switch (ectx->date_parse_order) {
  case date_parse_no_ambig:
    return PyUnicode_FromString("NoAmbig");
  case date_parse_ymd:
    return PyUnicode_FromString("YMD");
  case date_parse_dmy:
    return PyUnicode_FromString("DMY");
  case date_parse_mdy:
    return PyUnicode_FromString("MDY");
  default:
    throw invalid_argument("dynd internal error: invalid "
                           "date_parse_order in eval_context object");
  }
}

PyObject *pydynd::get_eval_context_century_window(PyObject *ectx_obj)
{
  if (!WEvalContext_Check(ectx_obj)) {
    throw invalid_argument("expected an nd.eval_context object");
  }
  const eval::eval_context *ectx = ((WEvalContext *)ectx_obj)->ectx;
  return PyLong_FromLong(ectx->century_window);
}

PyObject *pydynd::get_eval_context_repr(PyObject *ectx_obj)
{
  if (!WEvalContext_Check(ectx_obj)) {
    throw invalid_argument("expected an nd.eval_context object");
  }
  const eval::eval_context *ectx = ((WEvalContext *)ectx_obj)->ectx;
  stringstream ss;
  ss << "nd.eval_context(errmode='" << ectx->errmode << "',\n";
  ss << "                cuda_device_errmode='" << ectx->cuda_device_errmode
     << "',\n";
  ss << "                date_parse_order='" << ectx->date_parse_order
     << "',\n";
  ss << "                century_window=" << ectx->century_window << ")";
#if PY_VERSION_HEX < 0x03000000
  return PyString_FromString(ss.str().c_str());
#else
  return PyUnicode_FromString(ss.str().c_str());
#endif
}

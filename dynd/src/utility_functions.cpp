//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>

#include "utility_functions.hpp"
#include "type_functions.hpp"
#include "array_functions.hpp"

#include <dynd/exceptions.hpp>
#include <dynd/array.hpp>
#include <dynd/callable.hpp>

using namespace std;
using namespace dynd;
using namespace pydynd;

int pydynd::pyarg_strings_to_int(PyObject *obj, const char *argname,
                                 int default_value, const char *string0,
                                 int value0, const char *string1, int value1,
                                 const char *string2, int value2)
{
  if (obj == NULL || obj == Py_None) {
    return default_value;
  }

  std::string s = pystring_as_string(obj);

  if (s == string0) {
    return value0;
  } else if (s == string1) {
    return value1;
  } else if (s == string2) {
    return value2;
  }

  stringstream ss;
  ss << "argument " << argname << " was given the invalid argument value \""
     << s << "\"";
  throw runtime_error(ss.str());
}

bool pydynd::pyarg_bool(PyObject *obj, const char *argname, bool default_value)
{
  if (obj == NULL || obj == Py_None) {
    return default_value;
  }

  if (obj == Py_False) {
    return false;
  } else if (obj == Py_True) {
    return true;
  } else {
    stringstream ss;
    ss << "argument " << argname << " must be a boolean True or False";
    throw runtime_error(ss.str());
  }
}

uint32_t pydynd::pyarg_access_flags(PyObject *obj)
{
  pyobject_ownref iterator(PyObject_GetIter(obj));
  PyObject *item_raw;

  uint32_t result = 0;

  while ((item_raw = PyIter_Next(iterator))) {
    pyobject_ownref item(item_raw);
    result |= (uint32_t)pyarg_strings_to_int(
        item, "access_flags", 0, "read", nd::read_access_flag, "write",
        nd::write_access_flag, "immutable", nd::immutable_access_flag);
  }

  if (PyErr_Occurred()) {
    throw runtime_error("propagating exception...");
  }

  return result;
}

const dynd::callable_type_data *pydynd::pyarg_callable_ro(PyObject *af,
                                                          const char *paramname)
{
  if (!DyND_PyArray_Check(af) ||
      ((DyND_PyArrayObject *)af)->v.get_type().get_type_id() !=
          callable_type_id) {
    stringstream ss;
    ss << paramname << " must be an nd.array of type callable";
    throw runtime_error(ss.str());
  }
  return reinterpret_cast<const callable_type_data *>(
      ((DyND_PyArrayObject *)af)->v.cdata());
}

dynd::callable_type_data *pydynd::pyarg_callable_rw(PyObject *af,
                                                    const char *paramname)
{
  if (!DyND_PyArray_Check(af) ||
      ((DyND_PyArrayObject *)af)->v.get_type().get_type_id() !=
          callable_type_id) {
    stringstream ss;
    ss << paramname << " must be an nd.array of type callable";
    throw runtime_error(ss.str());
  }
  return reinterpret_cast<callable_type_data *>(
      ((DyND_PyArrayObject *)af)->v.data());
}

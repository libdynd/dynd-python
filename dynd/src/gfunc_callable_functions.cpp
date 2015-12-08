//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "gfunc_callable_functions.hpp"
#include "utility_functions.hpp"
#include "array_functions.hpp"
#include "array_as_py.hpp"
#include "array_assign_from_py.hpp"

#include <dynd/types/struct_type.hpp>
#include <dynd/types/builtin_type_properties.hpp>
#include <dynd/types/type_type.hpp>

using namespace std;
using namespace dynd;
using namespace pydynd;

PyTypeObject *pydynd::WArrayCallable_Type;

void pydynd::init_w_array_callable_typeobject(PyObject *type)
{
  WArrayCallable_Type = (PyTypeObject *)type;
}

PyTypeObject *pydynd::WTypeCallable_Type;

void pydynd::init_w__type_callable_typeobject(PyObject *type)
{
  WTypeCallable_Type = (PyTypeObject *)type;
}

void pydynd::add__type_names_to_dir_dict(const ndt::type &dt, PyObject *dict)
{
  if (!dt.is_builtin()) {
    const std::pair<std::string, nd::callable> *properties;
    size_t count;
    // Add the type properties
    dt.extended()->get_dynamic_type_properties(&properties, &count);
    for (size_t i = 0; i < count; ++i) {
      if (PyDict_SetItemString(dict, properties[i].first.c_str(), Py_None) <
          0) {
        throw runtime_error("");
      }
    }
    // Add the type functions
    dt.extended()->get_dynamic_type_functions(&properties, &count);
    for (size_t i = 0; i < count; ++i) {
      if (PyDict_SetItemString(dict, properties[i].first.c_str(), Py_None) <
          0) {
        throw runtime_error("");
      }
    }
  }
}

PyObject *pydynd::get__type_dynamic_property(const dynd::ndt::type &dt,
                                             PyObject *name)
{
  if (!dt.is_builtin()) {
    const std::pair<std::string, nd::callable> *properties;
    size_t count;
    // Search for a property
    dt.extended()->get_dynamic_type_properties(&properties, &count);
    // TODO: We probably want to make some kind of acceleration structure for
    // the name lookup
    if (count > 0) {
      std::string nstr = pystring_as_string(name);
      for (size_t i = 0; i < count; ++i) {
        if (properties[i].first == nstr) {
          return DyND_PyWrapper_New(const_cast<dynd::nd::callable &>(
              properties[i].second)(dynd::kwds("self", dt)));
          //          return call_gfunc_callable(nstr, properties[i].second,
          //          dt);
        }
      }
    }
    // Search for a function
    dt.extended()->get_dynamic_type_functions(&properties, &count);
    if (count > 0) {
      std::string nstr = pystring_as_string(name);
      for (size_t i = 0; i < count; ++i) {
        if (properties[i].first == nstr) {
          return wrap__type_callable(nstr, properties[i].second, dt);
        }
      }
    }
  }

  PyErr_SetObject(PyExc_AttributeError, name);
  return NULL;
}

void pydynd::add_array_names_to_dir_dict(const dynd::nd::array &n,
                                         PyObject *dict)
{
  ndt::type dt = n.get_type();
  if (!dt.is_builtin()) {
    const std::pair<std::string, nd::callable> *properties;
    size_t count;
    // Add the array properties
    dt.extended()->get_dynamic_array_properties(&properties, &count);
    for (size_t i = 0; i < count; ++i) {
      if (PyDict_SetItemString(dict, properties[i].first.c_str(), Py_None) <
          0) {
        throw runtime_error("");
      }
    }
    // Add the array functions
    std::map<std::string, nd::callable> functions;
    dt.extended()->get_dynamic_array_functions(functions);
    for (const auto &pair : functions)
      if (PyDict_SetItemString(dict, pair.first.c_str(), Py_None) < 0) {
        throw runtime_error("");
      }
  }
  else {
    const std::pair<std::string, nd::callable> *properties;
    size_t count;
    // Add the array properties
    get_builtin_type_dynamic_array_properties(dt.get_type_id(), &properties,
                                              &count);
    for (size_t i = 0; i < count; ++i) {
      if (PyDict_SetItemString(dict, properties[i].first.c_str(), Py_None) <
          0) {
        throw runtime_error("");
      }
    }
    // TODO: Add the array functions
  }
}

PyObject *pydynd::get_array_dynamic_property(const dynd::nd::array &n,
                                             PyObject *name)
{
  if (n.is_null()) {
    PyErr_SetObject(PyExc_AttributeError, name);
    return NULL;
  }
  ndt::type dt = n.get_type();
  const std::pair<std::string, nd::callable> *properties;
  size_t count;
  // Search for a property
  if (!dt.is_builtin()) {
    dt.extended()->get_dynamic_array_properties(&properties, &count);
  }
  else {
    get_builtin_type_dynamic_array_properties(dt.get_type_id(), &properties,
                                              &count);
  }

  // TODO: We probably want to make some kind of acceleration structure for the
  // name lookup
  if (count > 0) {
    std::string nstr = pystring_as_string(name);
    for (size_t i = 0; i < count; ++i) {
      if (properties[i].first == nstr) {
        return DyND_PyWrapper_New(const_cast<dynd::nd::callable &>(
            properties[i].second)(dynd::kwds("self", n)));
      }
    }
  }

  // Search for a function
  std::map<std::string, nd::callable> functions;
  if (!dt.is_builtin()) {
    dt.extended()->get_dynamic_array_functions(functions);
  }
  else {
    count = 0;
  }
  std::string nstr = pystring_as_string(name);
  nd::callable c = functions[nstr];
  if (!c.is_null()) {
    return DyND_PyWrapper_New(c(n));
  }

  PyErr_SetObject(PyExc_AttributeError, name);
  return NULL;
}

void pydynd::set_array_dynamic_property(const dynd::nd::array &n,
                                        PyObject *name, PyObject *value)
{
  ndt::type dt = n.get_type();
  const std::pair<std::string, nd::callable> *properties;
  size_t count;
  // Search for a property
  if (!dt.is_builtin()) {
    dt.extended()->get_dynamic_array_properties(&properties, &count);
  }
  else {
    get_builtin_type_dynamic_array_properties(dt.get_type_id(), &properties,
                                              &count);
  }
  // TODO: We probably want to make some kind of acceleration structure for
  // the name lookup
  if (count > 0) {
    std::string nstr = pystring_as_string(name);
    for (size_t i = 0; i < count; ++i) {
      if (properties[i].first == nstr) {
        nd::array p = const_cast<dynd::nd::callable &>(properties[i].second)(
            dynd::kwds("self", n));
        array_broadcast_assign_from_py(p, value, &eval::default_eval_context);
        return;
      }
    }
  }

  PyErr_SetObject(PyExc_AttributeError, name);
  throw exception();
}

static void set_single_parameter(const std::string &funcname,
                                 const std::string &paramname,
                                 const ndt::type &paramtype, char *arrmeta,
                                 char *data, const ndt::type &value)
{
  if (paramtype.get_type_id() != type_type_id) {
    stringstream ss;
    ss << "parameter \"" << paramname << "\" of dynd callable \"" << funcname
       << "\" with type " << paramtype;
    ss << " cannot accept a dynd type as its value";
    throw runtime_error(ss.str());
  }
  // The type is encoded as either a raw type id, or a pointer to a base_type,
  // just as the gfunc object is expecting.
  ndt::type(value).swap(*reinterpret_cast<dynd::ndt::type *>(data));
}

static void set_single_parameter(const std::string &funcname,
                                 const std::string &paramname,
                                 const ndt::type &paramtype, char *arrmeta,
                                 char *data, const nd::array &value)
{
  // NOTE: ndarrayarg is a borrowed reference to an nd::array
  if (paramtype.get_type_id() != ndarrayarg_type_id) {
    stringstream ss;
    ss << "parameter \"" << paramname << "\" of dynd callable \"" << funcname
       << "\" with type " << paramtype;
    ss << " cannot accept an array as its value";
    throw runtime_error(ss.str());
  }
  *(const void **)data = value.get();
}

/**
 * This converts a single PyObject input parameter into the requested dynd
 * parameter data.
 *
 * \param out_storage  This is a hack because dynd doesn't support object
 * lifetime management
 */
static void set_single_parameter(const ndt::type &paramtype, char *arrmeta,
                                 char *data, PyObject *value,
                                 vector<nd::array> &out_storage)
{
  // NOTE: ndarrayarg is a borrowed reference to an nd::array
  if (paramtype.get_type_id() == ndarrayarg_type_id) {
    out_storage.push_back(
        array_from_py(value, 0, false, &eval::default_eval_context));
    *(const void **)data = out_storage.back().get();
  }
  else {
    array_no_dim_broadcast_assign_from_py(paramtype, arrmeta, data, value,
                                          &eval::default_eval_context);
  }
}

PyObject *pydynd::wrap__type_callable(const std::string &funcname,
                                      const dynd::nd::callable &c,
                                      const dynd::ndt::type &d)
{
  WTypeCallable *result =
      (WTypeCallable *)WTypeCallable_Type->tp_alloc(WTypeCallable_Type, 0);
  if (!result) {
    return NULL;
  }
  // Calling tp_alloc doesn't call Cython's __cinit__, so do the placement new
  // here

  new (&result->v) _type_callable_wrapper();
  result->v.d = d;
  result->v.c = c;
  result->v.funcname = funcname;
  return (PyObject *)result;
}

static PyObject *_type_callable_call(const std::string &funcname,
                                     const nd::callable &c, const ndt::type &d,
                                     PyObject *args, PyObject *kwargs)
{
  return NULL;

  // Convert kwds into nd::arrays
  intptr_t nkwd = PyDict_Size(kwargs);
  vector<std::string> kwd_names_strings(nkwd);
  std::vector<const char *> kwd_names(nkwd);
  std::vector<dynd::nd::array> kwd_values(nkwd);
  PyObject *key, *value;
  for (Py_ssize_t i = 0, j = 0; PyDict_Next(kwargs, &i, &key, &value); ++j) {
    kwd_names_strings[j] = pystring_as_string(key);
    kwd_names[j] = kwd_names_strings[j].c_str();
    kwd_values[j] =
        array_from_py(value, 0, false, &dynd::eval::default_eval_context);
  }

  kwd_names_strings.insert(kwd_names_strings.begin(), "self");
  kwd_names.insert(kwd_names.begin(), "self");
  kwd_values.insert(kwd_values.begin(), d);

  dynd::nd::array result = const_cast<nd::callable &>(c)(
      0, static_cast<nd::array *>(NULL),
      kwds(nkwd + 1, kwd_names.empty() ? NULL : kwd_names.data(),
           kwd_values.empty() ? NULL : kwd_values.data()));
  return DyND_PyWrapper_New(result);

  /*
    const ndt::type &pdt = c.get_parameters_type();
    vector<nd::array> storage;
    nd::array params = nd::empty(pdt);
    const ndt::struct_type *fsdt = pdt.extended<ndt::struct_type>();
    // Set the 'self' parameter value
    set_single_parameter(funcname, fsdt->get_field_name(0),
                         fsdt->get_field_type(0),
                         params.get()->metadata() + fsdt->get_arrmeta_offset(0),
                         params.get()->data +
                             fsdt->get_data_offsets(params.get()->metadata())[0],
                         d);

    fill_thiscall_parameters_array(funcname, c, args, kwargs, params, storage);

    return DyND_PyWrapper_New(c.call_generic(params));
  */

  return NULL;
}

PyObject *pydynd::_type_callable_call(const _type_callable_wrapper &dcw,
                                      PyObject *args, PyObject *kwargs)
{
  return _type_callable_call(dcw.funcname, dcw.c, dcw.d, args, kwargs);
}

PyObject *pydynd::call__type_constructor_function(const dynd::ndt::type &dt,
                                                  PyObject *args,
                                                  PyObject *kwargs)
{
  // First find the __construct__ callable
  if (!dt.is_builtin()) {
    const std::pair<std::string, nd::callable> *properties;
    size_t count;
    // Search for a function
    dt.extended()->get_dynamic_type_functions(&properties, &count);
    if (count > 0) {
      for (size_t i = 0; i < count; ++i) {
        return _type_callable_call("self", properties[i].second, dt, args,
                                   kwargs);
      }
    }
  }

  stringstream ss;
  ss << "dynd type " << dt << " has no array constructor function";
  PyErr_SetString(PyExc_TypeError, ss.str().c_str());
  return NULL;
}

//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <Python.h>

#include <sstream>

#include <dynd/shape_tools.hpp>
#include <dynd/string_encodings.hpp>
#include <dynd/type.hpp>

#include <dynd/types/bytes_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/type_type.hpp>

#include "utility_functions.hpp"
#include "type_conversions.hpp"
#include "visibility.hpp"


// Python's datetime C API
#include "datetime.h"


namespace pydynd {

inline std::string _type_str(const dynd::ndt::type &d)
{
  std::stringstream ss;
  ss << d;
  return ss.str();
}

inline std::string _type_repr(const dynd::ndt::type &d)
{
  std::stringstream ss;
  ss << "ndt.type('" << d << "')";
  return ss.str();
}

inline PyObject *_type_get_shape(const dynd::ndt::type &d)
{
  size_t ndim = d.get_ndim();
  if (ndim > 0) {
    dynd::dimvector shape(ndim);
    d.extended()->get_shape(ndim, 0, shape.get(), NULL, NULL);
    return intptr_array_as_tuple(ndim, shape.get());
  }
  else {
    return PyTuple_New(0);
  }
}

inline dynd::string_encoding_t encoding_from_pyobject(PyObject *encoding_obj)
{
  // Default is utf-8
  if (encoding_obj == Py_None) {
    return dynd::string_encoding_utf_8;
  }

  dynd::string_encoding_t encoding = dynd::string_encoding_invalid;
  std::string encoding_str = pystring_as_string(encoding_obj);
  switch (encoding_str.size()) {
  case 4:
    switch (encoding_str[3]) {
    case '2':
      if (encoding_str == "ucs2") {
        encoding = dynd::string_encoding_ucs_2;
      }
      break;
    case '8':
      if (encoding_str == "utf8") {
        encoding = dynd::string_encoding_utf_8;
      }
      break;
    }
  case 5:
    switch (encoding_str[1]) {
    case 'c':
      if (encoding_str == "ucs_2" || encoding_str == "ucs-2") {
        encoding = dynd::string_encoding_ucs_2;
      }
      break;
    case 's':
      if (encoding_str == "ascii") {
        encoding = dynd::string_encoding_ascii;
      }
      break;
    case 't':
      if (encoding_str == "utf16") {
        encoding = dynd::string_encoding_utf_16;
      }
      else if (encoding_str == "utf32") {
        encoding = dynd::string_encoding_utf_32;
      }
      else if (encoding_str == "utf_8" || encoding_str == "utf-8") {
        encoding = dynd::string_encoding_utf_8;
      }
      break;
    }
    break;
  case 6:
    switch (encoding_str[4]) {
    case '1':
      if (encoding_str == "utf_16" || encoding_str == "utf-16") {
        encoding = dynd::string_encoding_utf_16;
      }
      break;
    case '3':
      if (encoding_str == "utf_32" || encoding_str == "utf-32") {
        encoding = dynd::string_encoding_utf_32;
      }
      break;
    }
  }

  if (encoding != dynd::string_encoding_invalid) {
    return encoding;
  }
  else {
    std::stringstream ss;
    ss << "invalid input \"" << encoding_str << "\" for string encoding";
    throw std::runtime_error(ss.str());
  }
}

inline dynd::ndt::type dynd_make_fixed_string_type(intptr_t size,
                                                   PyObject *encoding_obj)
{
  dynd::string_encoding_t encoding = encoding_from_pyobject(encoding_obj);

  return dynd::ndt::fixed_string_type::make(size, encoding);
}

inline dynd::ndt::type dynd_make_string_type(PyObject *DYND_UNUSED(encoding_obj))
{
  // dynd::string_encoding_t encoding = encoding_from_pyobject(encoding_obj);

  return dynd::ndt::make_type<dynd::ndt::string_type>();
}

inline dynd::ndt::type dynd_make_pointer_type(const dynd::ndt::type &target_tp)
{
  return dynd::ndt::pointer_type::make(target_tp);
}

inline void
pyobject_as_vector__type(PyObject *list_of_types,
                         std::vector<dynd::ndt::type> &vector_of__types)
{
  Py_ssize_t size = PySequence_Size(list_of_types);
  vector_of__types.resize(size);
  for (Py_ssize_t i = 0; i < size; ++i) {
    pyobject_ownref item(PySequence_GetItem(list_of_types, i));
    vector_of__types[i] = dynd_ndt_as_cpp_type(item.get());
  }
}

inline dynd::ndt::type dynd_make_struct_type(PyObject *field_types,
                                             PyObject *field_names)
{
  std::vector<dynd::ndt::type> field_types_vec;
  std::vector<std::string> field_names_vec;
  pyobject_as_vector__type(field_types, field_types_vec);
  pyobject_as_vector_string(field_names, field_names_vec);
  if (field_types_vec.size() != field_names_vec.size()) {
    std::stringstream ss;
    ss << "creating a struct type requires that the number of types ";
    ss << field_types_vec.size() << " must equal the number of names ";
    ss << field_names_vec.size();
    throw std::invalid_argument(ss.str());
  }
  return dynd::ndt::struct_type::make(field_names_vec, field_types_vec);
}

inline dynd::ndt::type
dynd_make_fixed_dim_type(PyObject *shape, const dynd::ndt::type &element_tp)
{
  std::vector<intptr_t> shape_vec;
  pyobject_as_vector_intp(shape, shape_vec, true);
  return dynd::ndt::make_fixed_dim(shape_vec.size(), &shape_vec[0], element_tp);
}

inline void init_type_functions()
{
  // Initialize the pydatetime API
  PyDateTime_IMPORT;
}

} // namespace pydynd

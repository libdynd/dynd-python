//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "type_functions.hpp"
#include "array_functions.hpp"
#include "numpy_interop.hpp"
#include "ctypes_interop.hpp"
#include "utility_functions.hpp"

using namespace std;
using namespace dynd;
using namespace pydynd;

static string_encoding_t encoding_from_pyobject(PyObject *encoding_obj)
{
  // Default is utf-8
  if (encoding_obj == Py_None) {
    return string_encoding_utf_8;
  }

  string_encoding_t encoding = string_encoding_invalid;
  std::string encoding_str = pystring_as_string(encoding_obj);
  switch (encoding_str.size()) {
  case 4:
    switch (encoding_str[3]) {
    case '2':
      if (encoding_str == "ucs2") {
        encoding = string_encoding_ucs_2;
      }
      break;
    case '8':
      if (encoding_str == "utf8") {
        encoding = string_encoding_utf_8;
      }
      break;
    }
  case 5:
    switch (encoding_str[1]) {
    case 'c':
      if (encoding_str == "ucs_2" || encoding_str == "ucs-2") {
        encoding = string_encoding_ucs_2;
      }
      break;
    case 's':
      if (encoding_str == "ascii") {
        encoding = string_encoding_ascii;
      }
      break;
    case 't':
      if (encoding_str == "utf16") {
        encoding = string_encoding_utf_16;
      }
      else if (encoding_str == "utf32") {
        encoding = string_encoding_utf_32;
      }
      else if (encoding_str == "utf_8" || encoding_str == "utf-8") {
        encoding = string_encoding_utf_8;
      }
      break;
    }
    break;
  case 6:
    switch (encoding_str[4]) {
    case '1':
      if (encoding_str == "utf_16" || encoding_str == "utf-16") {
        encoding = string_encoding_utf_16;
      }
      break;
    case '3':
      if (encoding_str == "utf_32" || encoding_str == "utf-32") {
        encoding = string_encoding_utf_32;
      }
      break;
    }
  }

  if (encoding != string_encoding_invalid) {
    return encoding;
  }
  else {
    stringstream ss;
    ss << "invalid input \"" << encoding_str << "\" for string encoding";
    throw std::runtime_error(ss.str());
  }
}

dynd::ndt::type pydynd::dynd_make_convert_type(const dynd::ndt::type &to_tp,
                                               const dynd::ndt::type &from_tp)
{
  return ndt::convert_type::make(to_tp, from_tp);
}

dynd::ndt::type pydynd::dynd_make_view_type(const dynd::ndt::type &value_type,
                                            const dynd::ndt::type &operand_type)
{
  return ndt::view_type::make(value_type, operand_type);
}

dynd::ndt::type pydynd::dynd_make_fixed_string_type(intptr_t size,
                                                    PyObject *encoding_obj)
{
  string_encoding_t encoding = encoding_from_pyobject(encoding_obj);

  return ndt::fixed_string_type::make(size, encoding);
}

dynd::ndt::type pydynd::dynd_make_string_type(PyObject *encoding_obj)
{
  string_encoding_t encoding = encoding_from_pyobject(encoding_obj);

  return ndt::make_type<dynd::ndt::string_type>();
}

dynd::ndt::type pydynd::dynd_make_pointer_type(const ndt::type &target_tp)
{
  return ndt::pointer_type::make(target_tp);
}

dynd::ndt::type pydynd::dynd_make_struct_type(PyObject *field_types,
                                              PyObject *field_names)
{
  vector<ndt::type> field_types_vec;
  vector<std::string> field_names_vec;
  pyobject_as_vector__type(field_types, field_types_vec);
  pyobject_as_vector_string(field_names, field_names_vec);
  if (field_types_vec.size() != field_names_vec.size()) {
    stringstream ss;
    ss << "creating a struct type requires that the number of types ";
    ss << field_types_vec.size() << " must equal the number of names ";
    ss << field_names_vec.size();
    throw invalid_argument(ss.str());
  }
  return ndt::struct_type::make(field_names_vec, field_types_vec);
}

dynd::ndt::type
pydynd::dynd_make_fixed_dim_type(PyObject *shape,
                                 const dynd::ndt::type &element_tp)
{
  vector<intptr_t> shape_vec;
  pyobject_as_vector_intp(shape, shape_vec, true);
  return ndt::make_fixed_dim(shape_vec.size(), &shape_vec[0], element_tp);
}

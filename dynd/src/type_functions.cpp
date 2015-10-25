//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "type_functions.hpp"
#include "array_functions.hpp"
#include "numpy_interop.hpp"
#include "ctypes_interop.hpp"
#include "utility_functions.hpp"

#include <dynd/types/convert_type.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/pointer_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/time_type.hpp>
#include <dynd/types/datetime_type.hpp>
#include <dynd/types/type_type.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/types/builtin_type_properties.hpp>

// Python's datetime C API
#include "datetime.h"

using namespace std;
using namespace dynd;
using namespace pydynd;

void pydynd::init_type_functions()
{
  // Initialize the pydatetime API
  PyDateTime_IMPORT;
}

void pydynd::init_w_type_typeobject(PyObject *type)
{
  DyND_PyWrapper_Type<dynd::ndt::type>() = (PyTypeObject *)type;
}

inline void print_generic_type_repr(ostream &o, const ndt::type &d)
{
  stringstream sstmp;
  sstmp << d;
  o << "ndt.type(";
  print_escaped_utf8_string(o, sstmp.str(), false);
  o << ")";
}

std::string pydynd::_type_repr(const dynd::ndt::type &d)
{
  std::stringstream ss;
  if (d.is_builtin() && d.get_type_id() != dynd::complex_float32_type_id &&
      d.get_type_id() != dynd::complex_float64_type_id) {
    ss << "ndt." << d;
  } else {
    switch (d.get_type_id()) {
    case complex_float32_type_id:
      ss << "ndt.complex_float32";
      break;
    case complex_float64_type_id:
      ss << "ndt.complex_float64";
      break;
    case date_type_id:
      ss << "ndt.date";
      break;
    case time_type_id:
      if (d.extended<ndt::time_type>()->get_timezone() == tz_abstract) {
        ss << "ndt.time";
      } else {
        print_generic_type_repr(ss, d);
      }
      break;
    case datetime_type_id:
      if (d.extended<ndt::datetime_type>()->get_timezone() == tz_abstract) {
        ss << "ndt.datetime";
      } else if (d.extended<ndt::datetime_type>()->get_timezone() == tz_utc) {
        ss << "ndt.datetimeutc";
      } else {
        print_generic_type_repr(ss, d);
      }
      break;
    case bytes_type_id:
      if (d.extended<ndt::bytes_type>()->get_target_alignment() == 1) {
        ss << "ndt.bytes";
      } else {
        print_generic_type_repr(ss, d);
      }
      break;
    case string_type_id:
      if (d.extended<ndt::string_type>()->get_encoding() ==
          string_encoding_utf_8) {
        ss << "ndt.string";
      } else {
        print_generic_type_repr(ss, d);
      }
      break;
    default:
      print_generic_type_repr(ss, d);
      break;
    }
  }
  return ss.str();
}

PyObject *pydynd::_type_get_shape(const dynd::ndt::type &d)
{
  size_t ndim = d.get_ndim();
  if (ndim > 0) {
    dimvector shape(ndim);
    d.extended()->get_shape(ndim, 0, shape.get(), NULL, NULL);
    return intptr_array_as_tuple(ndim, shape.get());
  } else {
    return PyTuple_New(0);
  }
}

PyObject *pydynd::_type_get_kind(const dynd::ndt::type &d)
{
  stringstream ss;
  ss << d.get_kind();
  std::string s = ss.str();
#if PY_VERSION_HEX >= 0x03000000
  return PyUnicode_FromStringAndSize(s.data(), s.size());
#else
  return PyString_FromStringAndSize(s.data(), s.size());
#endif
}

PyObject *pydynd::_type_get_type_id(const dynd::ndt::type &d)
{
  stringstream ss;
  ss << d.get_type_id();
  std::string s = ss.str();
#if PY_VERSION_HEX >= 0x03000000
  return PyUnicode_FromStringAndSize(s.data(), s.size());
#else
  return PyString_FromStringAndSize(s.data(), s.size());
#endif
}

/**
 * Creates a dynd type out of typical Python typeobjects.
 */
static dynd::ndt::type make__type_from_pytypeobject(PyTypeObject *obj)
{
  if (obj == &PyBool_Type) {
    return ndt::type::make<bool1>();
#if PY_VERSION_HEX < 0x03000000
  } else if (obj == &PyInt_Type) {
    return ndt::type::make<int32_t>();
#endif
  } else if (obj == &PyLong_Type) {
    return ndt::type::make<int32_t>();
  } else if (obj == &PyFloat_Type) {
    return ndt::type::make<double>();
  } else if (obj == &PyComplex_Type) {
    return ndt::type::make<dynd::complex<double>>();
  } else if (obj == &PyUnicode_Type) {
    return ndt::string_type::make();
  } else if (obj == &PyByteArray_Type) {
    return ndt::bytes_type::make(1);
#if PY_VERSION_HEX >= 0x03000000
  } else if (obj == &PyBytes_Type) {
    return ndt::bytes_type::make(1);
#else
  } else if (obj == &PyString_Type) {
    return ndt::string_type::make();
#endif
  } else if (PyObject_IsSubclass((PyObject *)obj, ctypes.PyCData_Type)) {
    // CTypes type object
    return _type_from_ctypes_cdatatype((PyObject *)obj);
  } else if (obj == PyDateTimeAPI->DateType) {
    return ndt::date_type::make();
  } else if (obj == PyDateTimeAPI->TimeType) {
    return ndt::time_type::make();
  } else if (obj == PyDateTimeAPI->DateTimeType) {
    return ndt::datetime_type::make();
  }

  stringstream ss;
  ss << "could not convert the Python TypeObject ";
  pyobject_ownref obj_repr(PyObject_Repr((PyObject *)obj));
  ss << pystring_as_string(obj_repr.get());
  ss << " into a dynd type";
  throw dynd::type_error(ss.str());
}

dynd::ndt::type pydynd::make__type_from_pyobject(PyObject *obj)
{
  if (DyND_PyType_Check(obj)) {
    return ((DyND_PyTypeObject *)obj)->v;
#if PY_VERSION_HEX < 0x03000000
  } else if (PyString_Check(obj)) {
    return ndt::type(pystring_as_string(obj));
  } else if (PyInt_Check(obj)) {
    return ndt::type(static_cast<type_id_t>(PyInt_AS_LONG(obj)));
#endif
  } else if (PyLong_Check(obj)) {
    return ndt::type(static_cast<type_id_t>(PyLong_AsLong(obj)));
  } else if (PyUnicode_Check(obj)) {
    return ndt::type(pystring_as_string(obj));
  } else if (DyND_PyArray_Check(obj)) {
    return ((DyND_PyArrayObject *)obj)->v.as<ndt::type>();
  } else if (PyType_Check(obj)) {
#if DYND_NUMPY_INTEROP
    ndt::type result;
    if (_type_from_numpy_scalar_typeobject((PyTypeObject *)obj, result) == 0) {
      return result;
    }
#endif // DYND_NUMPY_INTEROP
    return make__type_from_pytypeobject((PyTypeObject *)obj);
  }

#if DYND_NUMPY_INTEROP
  if (PyArray_DescrCheck(obj)) {
    return _type_from_numpy_dtype((PyArray_Descr *)obj);
  }
#endif // DYND_NUMPY_INTEROP

  stringstream ss;
  ss << "could not convert the object ";
  pyobject_ownref repr(PyObject_Repr(obj));
  ss << pystring_as_string(repr.get());
  ss << " into a dynd type";
  throw dynd::type_error(ss.str());
}

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
      } else if (encoding_str == "utf32") {
        encoding = string_encoding_utf_32;
      } else if (encoding_str == "utf_8" || encoding_str == "utf-8") {
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
  } else {
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

  return ndt::string_type::make();
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

dynd::ndt::type pydynd::_type_getitem(const dynd::ndt::type &d,
                                      PyObject *subscript)
{
  // Convert the pyobject into an array of iranges
  intptr_t size;
  shortvector<irange> indices;
  if (!PyTuple_Check(subscript)) {
    // A single subscript
    size = 1;
    indices.init(1);
    indices[0] = pyobject_as_irange(subscript);
  } else {
    size = PyTuple_GET_SIZE(subscript);
    // Tuple of subscripts
    indices.init(size);
    for (Py_ssize_t i = 0; i < size; ++i) {
      indices[i] = pyobject_as_irange(PyTuple_GET_ITEM(subscript, i));
    }
  }

  // Do an indexing operation
  return d.at_array((int)size, indices.get());
}

PyObject *pydynd::_type_array_property_names(const ndt::type &d)
{
  const std::pair<std::string, gfunc::callable> *properties;
  size_t count;
  if (!d.is_builtin()) {
    d.extended()->get_dynamic_array_properties(&properties, &count);
  } else {
    get_builtin_type_dynamic_array_properties(d.get_type_id(), &properties,
                                              &count);
  }
  pyobject_ownref result(PyList_New(count));
  for (size_t i = 0; i != count; ++i) {
    const std::string &s = properties[i].first;
#if PY_VERSION_HEX >= 0x03000000
    pyobject_ownref str_obj(PyUnicode_FromStringAndSize(s.data(), s.size()));
#else
    pyobject_ownref str_obj(PyString_FromStringAndSize(s.data(), s.size()));
#endif
    PyList_SET_ITEM(result.get(), i, str_obj.release());
  }
  return result.release();
}

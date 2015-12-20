//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <Python.h>

#include <sstream>

#include <dynd/type.hpp>
#include <dynd/string_encodings.hpp>
#include <dynd/shape_tools.hpp>

#include "visibility.hpp"
#include "wrapper.hpp"
#include "utility_functions.hpp"
#include "numpy_interop.hpp"
#include "ctypes_interop.hpp"

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

// Python's datetime C API
#include "datetime.h"

typedef DyND_PyWrapperObject<dynd::ndt::type> DyND_PyTypeObject;

inline int DyND_PyType_Check(PyObject *obj)
{
  return DyND_PyWrapper_Check<dynd::ndt::type>(obj);
}

namespace pydynd {

inline std::string _type_str(const dynd::ndt::type &d)
{
  std::stringstream ss;
  ss << d;
  return ss.str();
}

inline std::string _type_repr(const dynd::ndt::type &d) {
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

/**
 * Returns the type id of the dynd::ndt::type, as a Python string.
 */
PYDYND_API PyObject *_type_get_type_id(const dynd::ndt::type &d);

inline dynd::ndt::type make__type_from_pytypeobject(PyTypeObject *obj)
{
  if (obj == &PyBool_Type) {
    return dynd::ndt::make_type<dynd::bool1>();
#if PY_VERSION_HEX < 0x03000000
  }
  else if (obj == &PyInt_Type) {
    return dynd::ndt::make_type<int32_t>();
#endif
  }
  else if (obj == &PyLong_Type) {
    return dynd::ndt::make_type<int32_t>();
  }
  else if (obj == &PyFloat_Type) {
    return dynd::ndt::make_type<double>();
  }
  else if (obj == &PyComplex_Type) {
    return dynd::ndt::make_type<dynd::complex<double>>();
  }
  else if (obj == &PyUnicode_Type) {
    return dynd::ndt::make_type<dynd::ndt::string_type>();
  }
  else if (obj == &PyByteArray_Type) {
    return dynd::ndt::bytes_type::make(1);
#if PY_VERSION_HEX >= 0x03000000
  }
  else if (obj == &PyBytes_Type) {
    return dynd::ndt::bytes_type::make(1);
#else
  }
  else if (obj == &PyString_Type) {
    return dynd::ndt::make_type<dynd::ndt::string_type>();
#endif
  }
  else if (PyObject_IsSubclass((PyObject *)obj, ctypes.PyCData_Type)) {
    // CTypes type object
    return _type_from_ctypes_cdatatype((PyObject *)obj);
  }
  else if (obj == PyDateTimeAPI->DateType) {
    return dynd::ndt::date_type::make();
  }
  else if (obj == PyDateTimeAPI->TimeType) {
    return dynd::ndt::time_type::make();
  }
  else if (obj == PyDateTimeAPI->DateTimeType) {
    return dynd::ndt::datetime_type::make();
  }

  std::stringstream ss;
  ss << "could not convert the Python TypeObject ";
  pyobject_ownref obj_repr(PyObject_Repr((PyObject *)obj));
  ss << pystring_as_string(obj_repr.get());
  ss << " into a dynd type";
  throw dynd::type_error(ss.str());
}

/**
 * Converts a Python type, numpy dtype, or string
 * into an dynd::ndt::type. This raises an error if given an object
 * which contains values, it is for type-like things only.
 */
 inline dynd::ndt::type make__type_from_pyobject(PyObject *obj)
 {
   if (DyND_PyType_Check(obj)) {
     return ((DyND_PyTypeObject *)obj)->v;
 #if PY_VERSION_HEX < 0x03000000
   }
   else if (PyString_Check(obj)) {
     return dynd::ndt::type(pystring_as_string(obj));
   }
   else if (PyInt_Check(obj)) {
     return dynd::ndt::type(static_cast<type_id_t>(PyInt_AS_LONG(obj)));
 #endif
   }
   else if (PyLong_Check(obj)) {
     return dynd::ndt::type(static_cast<dynd::type_id_t>(PyLong_AsLong(obj)));
   }
   else if (PyUnicode_Check(obj)) {
     return dynd::ndt::type(pystring_as_string(obj));
   }
   else if (DyND_PyArray_Check(obj)) {
     return ((DyND_PyArrayObject *)obj)->v.as<dynd::ndt::type>();
   }
   else if (PyType_Check(obj)) {
 #if DYND_NUMPY_INTEROP
     dynd::ndt::type result;
     if (_type_from_numpy_scalar_typeobject((PyTypeObject *)obj, result) == 0) {
       return result;
     }
 #endif // DYND_NUMPY_INTEROP
     return make__type_from_pytypeobject((PyTypeObject *)obj);
   }

 #if DYND_NUMPY_INTEROP
   if (is_numpy_dtype(obj)) {
     return _type_from_numpy_dtype((PyArray_Descr *)obj);
   }
 #endif // DYND_NUMPY_INTEROP

   std::stringstream ss;
   ss << "could not convert the object ";
   pyobject_ownref repr(PyObject_Repr(obj));
   ss << pystring_as_string(repr.get());
   ss << " into a dynd type";
   throw dynd::type_error(ss.str());
 }

/**
 * Creates a convert type.
 */
PYDYND_API dynd::ndt::type
dynd_make_convert_type(const dynd::ndt::type &to_tp,
                       const dynd::ndt::type &from_tp);

/**
 * Creates a view type.
 */
PYDYND_API dynd::ndt::type
dynd_make_view_type(const dynd::ndt::type &value_type,
                    const dynd::ndt::type &operand_type);

/**
 * Creates a fixed-sized string type.
 */
PYDYND_API dynd::ndt::type dynd_make_fixed_string_type(intptr_t size,
                                                       PyObject *encoding_obj);

/**
 * Creates a blockref string type.
 */
PYDYND_API dynd::ndt::type dynd_make_string_type(PyObject *encoding_obj);

/**
 * Creates a blockref pointer type.
 */
PYDYND_API dynd::ndt::type
dynd_make_pointer_type(const dynd::ndt::type &target_tp);

PYDYND_API dynd::ndt::type dynd_make_struct_type(PyObject *field_types,
                                                 PyObject *field_names);

PYDYND_API dynd::ndt::type
dynd_make_fixed_dim_type(PyObject *shape, const dynd::ndt::type &element_tp);

inline void init_type_functions()
{
  // Initialize the pydatetime API
  PyDateTime_IMPORT;
}

} // namespace pydynd

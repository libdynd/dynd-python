//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <Python.h>

#include <sstream>

#include <dynd/type.hpp>
#include <dynd/string_encodings.hpp>

#include "config.hpp"
#include "wrapper.hpp"

typedef DyND_PyWrapperObject<dynd::ndt::type> DyND_PyTypeObject;

namespace pydynd {

/**
 * This is the typeobject and struct of w_type from Cython.
 */
extern PyTypeObject *WType_Type;
inline bool WType_CheckExact(PyObject *obj)
{
  return Py_TYPE(obj) == WType_Type;
}
inline bool WType_Check(PyObject *obj)
{
  return PyObject_TypeCheck(obj, WType_Type);
}

PYDYND_API void init_w_type_typeobject(PyObject *type);

inline PyObject *wrap__type(const dynd::ndt::type &d)
{
  DyND_PyTypeObject *result = (DyND_PyTypeObject *)WType_Type->tp_alloc(WType_Type, 0);
  if (!result) {
    throw std::runtime_error("");
  }

  // Calling tp_alloc doesn't call Cython's __cinit__, so do the placement new
  // here
  new (&result->v) dynd::ndt::type(d);
  return (PyObject *)result;
}

#ifdef DYND_RVALUE_REFS
inline PyObject *wrap__type(dynd::ndt::type &&d)
{
  DyND_PyTypeObject *result = (DyND_PyTypeObject *)WType_Type->tp_alloc(WType_Type, 0);
  if (!result) {
    throw std::runtime_error("");
  }
  // Calling tp_alloc doesn't call Cython's __cinit__, so do the placement new
  // here
  pydynd::placement_new(
      reinterpret_cast<pydynd::_type_placement_wrapper &>(result->v));
  result->v = DYND_MOVE(d);
  return (PyObject *)result;
}
#endif

inline std::string _type_str(const dynd::ndt::type &d)
{
  std::stringstream ss;
  ss << d;
  return ss.str();
}

PYDYND_API std::string _type_repr(const dynd::ndt::type &d);

PYDYND_API PyObject *_type_get_shape(const dynd::ndt::type &d);

/**
 * Returns the kind of the ndt::type, as a Python string.
 */
PYDYND_API PyObject *_type_get_kind(const dynd::ndt::type &d);

/**
 * Returns the type id of the ndt::type, as a Python string.
 */
PYDYND_API PyObject *_type_get_type_id(const dynd::ndt::type &d);

/**
 * Converts a Python type, numpy dtype, or string
 * into an ndt::type. This raises an error if given an object
 * which contains values, it is for type-like things only.
 */
PYDYND_API dynd::ndt::type make__type_from_pyobject(PyObject *obj);

/**
 * Creates a convert type.
 */
PYDYND_API dynd::ndt::type dynd_make_convert_type(const dynd::ndt::type &to_tp,
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

/**
 * Implementation of __getitem__ for the wrapped dynd type object.
 */
PYDYND_API dynd::ndt::type _type_getitem(const dynd::ndt::type &d,
                                         PyObject *subscript);

PYDYND_API PyObject *_type_array_property_names(const dynd::ndt::type &d);

PYDYND_API void init_type_functions();

} // namespace pydynd
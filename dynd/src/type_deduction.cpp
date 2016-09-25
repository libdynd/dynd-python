//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>
#include <datetime.h>

#include <dynd/callable.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/type_promotion.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/type_type.hpp>
#include <dynd/types/var_dim_type.hpp>

#include "type_deduction.hpp"
#include "type_functions.hpp"

using namespace std;
using namespace dynd;
using namespace pydynd;

size_t pydynd::get_nonragged_dim_count(const ndt::type &tp, size_t max_count)
{
  if (tp.is_symbolic()) {
    if (tp.is_scalar()) {
      return 0;
    }
  }

  if (!tp.is_scalar()) {
    if (max_count <= 1) {
      return max_count;
    }
    else {
      return min(max_count,
                 1 + get_nonragged_dim_count(static_cast<const ndt::base_dim_type *>(tp.extended())->get_element_type(),
                                             max_count - 1));
    }
  }

  switch (tp.get_id()) {
  case struct_id:
  case tuple_id:
    if (max_count <= 1) {
      return max_count;
    }
    else {
      auto bsd = tp.extended<ndt::tuple_type>();
      size_t field_count = bsd->get_field_count();
      for (size_t i = 0; i != field_count; ++i) {
        size_t candidate = 1 + get_nonragged_dim_count(bsd->get_field_type(i), max_count - 1);
        if (candidate < max_count) {
          max_count = candidate;
          if (max_count <= 1) {
            return max_count;
          }
        }
      }
      return max_count;
    }
  default:
    return 0;
  }
}

void pydynd::deduce_pyseq_shape(PyObject *obj, size_t ndim, intptr_t *shape)
{
  bool is_sequence = (PySequence_Check(obj) != 0);
  Py_ssize_t size = 0;
  if (is_sequence) {
    size = PySequence_Size(obj);
    if (size == -1 && PyErr_Occurred()) {
      PyErr_Clear();
      is_sequence = false;
    }
  }

  if (is_sequence) {
    if (shape[0] == pydynd_shape_deduction_uninitialized) {
      shape[0] = size;
    }
    else if (shape[0] != size) {
      // A variable-sized dimension
      shape[0] = pydynd_shape_deduction_var;
    }

    if (ndim > 1) {
      for (Py_ssize_t i = 0; i < size; ++i) {
        py_ref item = capture_if_not_null(PySequence_GetItem(obj, i));
        deduce_pyseq_shape(item.get(), ndim - 1, shape + 1);
      }
    }
  }
  else {
    // If it's an iterator, error checking needs to be done later
    // during actual value assignment.
    PyObject *iter = PyObject_GetIter(obj);
    if (iter == NULL) {
      if (PyErr_ExceptionMatches(PyExc_TypeError)) {
        PyErr_Clear();
        throw runtime_error("not enough dimensions in"
                            " python object for the provided dynd type");
      }
      else {
        // Propagate the exception
        throw exception();
      }
    }
    else {
      Py_DECREF(iter);
    }
    // It must be a variable-sized dimension
    shape[0] = pydynd_shape_deduction_var;
  }
}

void pydynd::deduce_pyseq_shape_using_dtype(PyObject *obj, const ndt::type &tp, std::vector<intptr_t> &shape,
                                            bool initial_pass, size_t current_axis)
{
  bool is_sequence = (PySequence_Check(obj) != 0 && !PyUnicode_Check(obj) && !PyDict_Check(obj));
#if PY_VERSION_HEX < 0x03000000
  is_sequence = is_sequence && !PyString_Check(obj);
#endif
  Py_ssize_t size = 0;
  if (is_sequence) {
    size = PySequence_Size(obj);
    if (size == -1 && PyErr_Occurred()) {
      PyErr_Clear();
      is_sequence = false;
    }
  }

  if (is_sequence) {
    if (shape.size() == current_axis) {
      if (initial_pass) {
        shape.push_back(size);
      }
      else if (tp.get_id() == struct_id || tp.get_id() == tuple_id) {
        // Signal that this is a dimension which is sometimes scalar, to allow
        // for
        // raggedness in the struct type's fields
        shape.push_back(pydynd_shape_deduction_ragged);
      }
      else {
        throw runtime_error("dynd array doesn't support dimensions"
                            " which are sometimes scalars and sometimes arrays");
      }
    }
    else {
      if (shape[current_axis] != size && shape[current_axis] >= 0) {
        // A variable-sized dimension
        shape[current_axis] = pydynd_shape_deduction_var;
      }
    }

    for (Py_ssize_t i = 0; i < size; ++i) {
      py_ref item = capture_if_not_null(PySequence_GetItem(obj, i));
      deduce_pyseq_shape_using_dtype(item.get(), tp, shape, i == 0 && initial_pass, current_axis + 1);
    }
  }
  else {
    if (PyDict_Check(obj) && tp.get_id() == struct_id) {
      if (shape.size() == current_axis) {
        shape.push_back(pydynd_shape_deduction_dict);
      }
      else if (shape[current_axis] != pydynd_shape_deduction_ragged) {
        shape[current_axis] = pydynd_shape_deduction_dict;
      }
    }
    else if (shape.size() != current_axis) {
      if (tp.get_id() == struct_id || tp.get_id() == tuple_id) {
        shape[current_axis] = pydynd_shape_deduction_ragged;
      }
      else {
        throw runtime_error("dynd array doesn't support dimensions"
                            " which are sometimes scalars and sometimes arrays");
      }
    }
  }
}

/**
 * Gets the number of dimensions at index 0, including tuple
 * and struct as dimensions.
 */
static intptr_t get_leading_dim_count(const dynd::ndt::type &tp)
{
  intptr_t ndim = tp.get_ndim();
  if (ndim) {
    return ndim + get_leading_dim_count(tp.get_dtype());
  }
  else if (tp.get_base_id() == expr_kind_id) {
    return get_leading_dim_count(tp.value_type());
  }
  else if (tp.get_id() == tuple_id || tp.get_id() == struct_id) {
    if (tp.extended<ndt::tuple_type>()->get_field_count() == 0) {
      return 1;
    }
    else {
      return 1 + get_leading_dim_count(tp.extended<ndt::tuple_type>()->get_field_type(0));
    }
  }
  else {
    return 0;
  }
}

bool pydynd::broadcast_as_scalar(const dynd::ndt::type &tp, PyObject *obj)
{
  intptr_t obj_ndim = 0;
  // Estimate the number of dimensions in ``obj`` by repeatedly indexing
  // along zero
  py_ref v = capture_if_not_null(obj);
  // incref the object since we just claimed there was a reference there to capture,
  // even though this function really just takes a borrowed reference.
  Py_INCREF(v.get());
  for (;;) {
    // Don't treat these types as sequences
    if (PyDict_Check(v.get())) {
      if (tp.get_dtype().get_id() == struct_id) {
        // If the object to assign to a dynd struct ends in a dict, apply
        // the dict as the struct's value
        return (tp.get_ndim() > obj_ndim);
      }
      break;
    }
    else if (PyUnicode_Check(v.get()) || PyBytes_Check(v.get())) {
      break;
    }
    PyObject *iter = PyObject_GetIter(v.get());
    if (iter != NULL) {
      ++obj_ndim;
      if (iter == v.get()) {
        // This was already an iterator, don't do any broadcasting,
        // because we have no visibility into it.
        Py_DECREF(iter);
        return false;
      }
      else {
        py_ref iter_owner = capture_if_not_null(iter);
        PyObject *item = PyIter_Next(iter);
        if (item == NULL) {
          if (PyErr_ExceptionMatches(PyExc_StopIteration)) {
            PyErr_Clear();
            break;
          }
          else {
            throw exception();
          }
        }
        else {
          v = capture_if_not_null(item);
        }
      }
    }
    else {
      PyErr_Clear();
      break;
    }
  }

  return (get_leading_dim_count(tp) > obj_ndim);
}

static PyTypeObject *array_pytypeobject = nullptr;
static dynd::ndt::type (*type_from_pyarray)(PyObject *) = nullptr;

void pydynd::register_nd_array_type_deduction(PyTypeObject *array_type, dynd::ndt::type (*get_type)(PyObject *))
{
  array_pytypeobject = array_type;
  type_from_pyarray = get_type;
}

dynd::ndt::type pydynd::xtype_for_prefix(PyObject *obj)
{
  // If it's a Cython w_array
  if (array_pytypeobject != nullptr) {
    if (PyObject_TypeCheck(obj, array_pytypeobject)) {
      return type_from_pyarray(obj);
    }
  }

#if DYND_NUMPY_INTEROP
  if (PyArray_Check(obj)) {
    return array_from_numpy_array2((PyArrayObject *)obj);
  }

#endif // DYND_NUMPY_INTEROP
  if (PyBool_Check(obj)) {
    return ndt::make_type<bool>();
  }
#if PY_VERSION_HEX < 0x03000000
  if (PyInt_Check(obj)) {
    long value = PyInt_AS_LONG(obj);
#if SIZEOF_LONG > SIZEOF_INT
    // Use a 32-bit int if it fits.
    if (value >= INT_MIN && value <= INT_MAX) {
      return ndt::make_type<int>();
    }
    else {
      return ndt::make_type<long>();
    }
#else
    return ndt::make_type<long>();
#endif
  }
#endif // PY_VERSION_HEX < 0x03000000
  if (PyLong_Check(obj)) {
    PY_LONG_LONG value = PyLong_AsLongLong(obj);
    if (value == -1 && PyErr_Occurred()) {
      throw runtime_error("error converting int value");
    }

    // Use a 32-bit int if it fits.
    if (value >= INT_MIN && value <= INT_MAX) {
      return ndt::make_type<int>();
    }
    else {
      return ndt::make_type<PY_LONG_LONG>();
    }
  }

  return dynd::ndt::type();
}

dynd::ndt::type pydynd::ndt_type_from_pylist(PyObject *obj)
{
  // TODO: Add ability to specify access flags (e.g. immutable)
  // Do a pass through all the data to deduce its type and shape
  std::vector<intptr_t> shape;
  dynd::ndt::type tp = dynd::ndt::make_type<void>();
  Py_ssize_t size = PyList_GET_SIZE(obj);
  shape.push_back(size);
  for (Py_ssize_t i = 0; i < size; ++i) {
    deduce_pylist_shape_and_dtype(PyList_GET_ITEM(obj, i), shape, tp, 1);
  }

  if (tp.get_id() == dynd::void_id) {
    tp = dynd::ndt::make_type<int32_t>();
  }

  return dynd::ndt::make_type(shape.size(), shape.data(), tp);
}

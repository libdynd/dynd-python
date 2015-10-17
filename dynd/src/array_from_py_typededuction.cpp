//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>
#include <datetime.h>

#include <dynd/func/callable.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/base_struct_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/time_type.hpp>
#include <dynd/types/datetime_type.hpp>
#include <dynd/types/type_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/type_promotion.hpp>
#include <dynd/exceptions.hpp>

#include "array_from_py_typededuction.hpp"
#include "array_assign_from_py.hpp"
#include "array_functions.hpp"
#include "type_functions.hpp"
#include "utility_functions.hpp"
#include "numpy_interop.hpp"

using namespace std;
using namespace dynd;
using namespace pydynd;

void pydynd::init_array_from_py_typededuction()
{
  // Initialize the pydatetime API
  PyDateTime_IMPORT;
}

ndt::type pydynd::deduce__type_from_pyobject(PyObject *obj,
                                                bool throw_on_unknown)
{
#if DYND_NUMPY_INTEROP
  if (PyArray_Check(obj)) {
    // Numpy array
    PyArray_Descr *d = PyArray_DESCR((PyArrayObject *)obj);
    return _type_from_numpy_dtype(d);
  }
  else if (PyArray_IsScalar(obj, Generic)) {
    // Numpy scalar
    return _type_of_numpy_scalar(obj);
  }
#endif // DYND_NUMPY_INTEROP

  if (PyBool_Check(obj)) {
    // Python bool
    return ndt::type::make<dynd::bool1>();
#if PY_VERSION_HEX < 0x03000000
  }
  else if (PyInt_Check(obj)) {
// Python integer
#if SIZEOF_LONG > SIZEOF_INT
    long value = PyInt_AS_LONG(obj);
    // Use a 32-bit int if it fits. This conversion strategy
    // is independent of sizeof(long), and is the same on 32-bit
    // and 64-bit platforms.
    if (value >= INT_MIN && value <= INT_MAX) {
      return ndt::type::make<int>();
    }
    else {
      return ndt::type::make<long>();
    }
#else
    return ndt::type::make<int>();
#endif
#endif // PY_VERSION_HEX < 0x03000000
  }
  else if (PyLong_Check(obj)) {
    // Python integer
    PY_LONG_LONG value = PyLong_AsLongLong(obj);
    if (value == -1 && PyErr_Occurred()) {
      throw runtime_error("error converting int value");
    }
    // Use a 32-bit int if it fits. This conversion strategy
    // is independent of sizeof(long), and is the same on 32-bit
    // and 64-bit platforms.
    if (value >= INT_MIN && value <= INT_MAX) {
      return ndt::type::make<int>();
    }
    else {
      return ndt::type::make<PY_LONG_LONG>();
    }
  }
  else if (PyFloat_Check(obj)) {
    // Python float
    return ndt::type::make<double>();
  }
  else if (PyComplex_Check(obj)) {
    // Python complex
    return ndt::type::make<dynd::complex<double>>();
#if PY_VERSION_HEX < 0x03000000
  }
  else if (PyString_Check(obj)) {
    // Python string
    return ndt::string_type::make();
#else
  }
  else if (PyBytes_Check(obj)) {
    // Python bytes string
    return ndt::bytes_type::make(1);
#endif
  }
  else if (PyUnicode_Check(obj)) {
    // Python string
    return ndt::string_type::make();
  }
  else if (PyDateTime_Check(obj)) {
    if (((PyDateTime_DateTime *)obj)->hastzinfo &&
        ((PyDateTime_DateTime *)obj)->tzinfo != NULL) {
      throw runtime_error("Converting datetimes with a timezone to dynd arrays "
                          "is not yet supported");
    }
    return ndt::datetime_type::make();
  }
  else if (PyDate_Check(obj)) {
    return ndt::date_type::make();
  }
  else if (PyTime_Check(obj)) {
    if (((PyDateTime_DateTime *)obj)->hastzinfo &&
        ((PyDateTime_DateTime *)obj)->tzinfo != NULL) {
      throw runtime_error("Converting times with a timezone to dynd arrays is "
                          "not yet supported");
    }
    return ndt::time_type::make(tz_abstract);
  }
  else if (DyND_PyType_Check(obj)) {
    return ndt::make_type();
  }
  else if (PyType_Check(obj)) {
    return ndt::make_type();
#if DYND_NUMPY_INTEROP
  }
  else if (PyArray_DescrCheck(obj)) {
    return ndt::make_type();
#endif // DYND_NUMPY_INTEROP
  }
  else if (obj == Py_None) {
    return ndt::option_type::make(ndt::type::make<void>());
  }

  // Check for a blaze.Array, or something which looks similar,
  // specifically named 'Array' and with a property 'dshape'
  PyObject *pytypename =
      PyObject_GetAttrString((PyObject *)Py_TYPE(obj), "__name__");
  if (pytypename != NULL) {
    pyobject_ownref pytypename_obj(pytypename);
    if (pystring_as_string(pytypename) == "Array") {
      PyObject *dshape = PyObject_GetAttrString(obj, "dshape");
      if (dshape != NULL) {
        pyobject_ownref dshape_obj(dshape);
        pyobject_ownref dshapestr_obj(PyObject_Str(dshape));
        return ndt::type(pystring_as_string(dshapestr_obj.get()));
      }
      else {
        PyErr_Clear();
      }
    }
  }
  else {
    PyErr_Clear();
  }

  if (throw_on_unknown) {
    stringstream ss;
    ss << "could not deduce pydynd type from the python object ";
    pyobject_ownref repr_obj(PyObject_Repr(obj));
    ss << pystring_as_string(repr_obj.get());
    throw std::runtime_error(ss.str());
  }
  else {
    // Return an uninitialized type to signal nothing was deduced
    return ndt::type();
  }
}

void pydynd::deduce_pylist_shape_and_dtype(PyObject *obj,
                                           vector<intptr_t> &shape,
                                           ndt::type &tp, size_t current_axis)
{
  if (PyList_Check(obj)) {
    Py_ssize_t size = PyList_GET_SIZE(obj);
    if (shape.size() == current_axis) {
      if (tp.get_type_id() == void_type_id) {
        shape.push_back(size);
      }
      else {
        throw runtime_error("dynd array doesn't support dimensions "
                            "which are sometimes scalars and sometimes arrays");
      }
    }
    else {
      if (shape[current_axis] != size) {
        // A variable-sized dimension
        shape[current_axis] = pydynd_shape_deduction_var;
      }
    }

    for (Py_ssize_t i = 0; i < size; ++i) {
      deduce_pylist_shape_and_dtype(PyList_GET_ITEM(obj, i), shape, tp,
                                    current_axis + 1);
      // Propagate uninitialized_type_id as a signal an
      // undeducable object was encountered
      if (tp.get_type_id() == uninitialized_type_id) {
        return;
      }
    }
  }
  else {
    if (shape.size() != current_axis) {
      // Return uninitialized_type_id as a signal
      // when an ambiguous situation like this is encountered,
      // letting the dynamic conversion handle it.
      tp = ndt::type();
      return;
    }

    ndt::type obj_tp;
#if PY_VERSION_HEX >= 0x03000000
    if (PyUnicode_Check(obj)) {
      obj_tp = ndt::string_type::make();
    }
    else {
      obj_tp = pydynd::deduce__type_from_pyobject(obj, false);
      // Propagate uninitialized_type_id as a signal an
      // undeducable object was encountered
      if (obj_tp.get_type_id() == uninitialized_type_id) {
        tp = obj_tp;
        return;
      }
    }
#else
    obj_tp = pydynd::deduce__type_from_pyobject(obj, false);
    // Propagate uninitialized_type_id as a signal an
    // undeducable object was encountered
    if (obj_tp.get_type_id() == uninitialized_type_id) {
      tp = obj_tp;
      return;
    }
#endif

    if (tp != obj_tp) {
      tp = dynd::promote_types_arithmetic(obj_tp, tp);
    }
  }
}

size_t pydynd::get_nonragged_dim_count(const ndt::type &tp, size_t max_count)
{
  switch (tp.get_kind()) {
  case kind_kind:
  case pattern_kind:
    if (tp.is_scalar()) {
      return 0;
    }
  case dim_kind:
    if (max_count <= 1) {
      return max_count;
    }
    else {
      return min(max_count, 1 + get_nonragged_dim_count(
                                    static_cast<const ndt::base_dim_type *>(
                                        tp.extended())->get_element_type(),
                                    max_count - 1));
    }
  case struct_kind:
  case tuple_kind:
    if (max_count <= 1) {
      return max_count;
    }
    else {
      auto bsd = tp.extended<ndt::base_tuple_type>();
      size_t field_count = bsd->get_field_count();
      for (size_t i = 0; i != field_count; ++i) {
        size_t candidate =
            1 + get_nonragged_dim_count(bsd->get_field_type(i), max_count - 1);
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
        pyobject_ownref item(PySequence_GetItem(obj, i));
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

void pydynd::deduce_pyseq_shape_using_dtype(PyObject *obj, const ndt::type &tp,
                                            std::vector<intptr_t> &shape,
                                            bool initial_pass,
                                            size_t current_axis)
{
  bool is_sequence = (PySequence_Check(obj) != 0 && !PyUnicode_Check(obj) &&
                      !PyDict_Check(obj));
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
      else if (tp.get_kind() == struct_kind || tp.get_kind() == tuple_kind) {
        // Signal that this is a dimension which is sometimes scalar, to allow
        // for
        // raggedness in the struct type's fields
        shape.push_back(pydynd_shape_deduction_ragged);
      }
      else {
        throw runtime_error(
            "dynd array doesn't support dimensions"
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
      pyobject_ownref item(PySequence_GetItem(obj, i));
      deduce_pyseq_shape_using_dtype(item.get(), tp, shape,
                                     i == 0 && initial_pass, current_axis + 1);
    }
  }
  else {
    if (PyDict_Check(obj) && tp.get_kind() == struct_kind) {
      if (shape.size() == current_axis) {
        shape.push_back(pydynd_shape_deduction_dict);
      }
      else if (shape[current_axis] != pydynd_shape_deduction_ragged) {
        shape[current_axis] = pydynd_shape_deduction_dict;
      }
    }
    else if (shape.size() != current_axis) {
      if (tp.get_kind() == struct_kind || tp.get_kind() == tuple_kind) {
        shape[current_axis] = pydynd_shape_deduction_ragged;
      }
      else {
        throw runtime_error(
            "dynd array doesn't support dimensions"
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
  else if (tp.get_kind() == expr_kind) {
    return get_leading_dim_count(tp.value_type());
  }
  else if (tp.get_kind() == tuple_kind || tp.get_kind() == struct_kind) {
    if (tp.extended<ndt::base_tuple_type>()->get_field_count() == 0) {
      return 1;
    }
    else {
      return 1 + get_leading_dim_count(
                     tp.extended<ndt::base_tuple_type>()->get_field_type(0));
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
  pyobject_ownref v(obj);
  Py_INCREF(v);
  for (;;) {
    // Don't treat these types as sequences
    if (PyDict_Check(v)) {
      if (tp.get_dtype().get_kind() == struct_kind) {
        // If the object to assign to a dynd struct ends in a dict, apply
        // the dict as the struct's value
        return (tp.get_ndim() > obj_ndim);
      }
      break;
    }
    else if (PyUnicode_Check(v) || PyBytes_Check(v)) {
      break;
    }
    PyObject *iter = PyObject_GetIter(v);
    if (iter != NULL) {
      ++obj_ndim;
      if (iter == v.get()) {
        // This was already an iterator, don't do any broadcasting,
        // because we have no visibility into it.
        Py_DECREF(iter);
        return false;
      }
      else {
        pyobject_ownref iter_owner(iter);
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
          v.reset(item);
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

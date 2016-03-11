//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <Python.h>

#include <sstream>
#include <stdexcept>
#include <string>

#include <dynd/type.hpp>

#include "visibility.hpp"

namespace dynd {
// Forward declaration
struct callable_type_data;
} // namespace dynd

namespace pydynd {

/**
 * A container class for managing the local lifetime of
 * PyObject *.
 *
 * Throws an exception if the object passed into the constructor
 * is NULL.
 */
class pyobject_ownref {
  PyObject *m_obj;

  // Non-copyable
  pyobject_ownref(const pyobject_ownref &);
  pyobject_ownref &operator=(const pyobject_ownref &);

public:
  inline pyobject_ownref() : m_obj(NULL) {}
  inline explicit pyobject_ownref(PyObject *obj) : m_obj(obj)
  {
    if (obj == NULL) {
      throw std::runtime_error("propagating a Python exception...");
    }
  }

  inline pyobject_ownref(PyObject *obj, bool inc_ref) : m_obj(obj)
  {
    if (obj == NULL) {
      throw std::runtime_error("propagating a Python exception...");
    }
    if (inc_ref) {
      Py_INCREF(m_obj);
    }
  }

  inline ~pyobject_ownref() { Py_XDECREF(m_obj); }

  inline PyObject **obj_addr() { return &m_obj; }

  /**
   * Resets the reference owned by this object to the one provided.
   * This steals a reference to the input parameter, 'obj'.
   *
   * \param obj  The reference to replace the current one in this object.
   */
  inline void reset(PyObject *obj)
  {
    if (obj == NULL) {
      throw std::runtime_error("propagating a Python exception...");
    }
    Py_XDECREF(m_obj);
    m_obj = obj;
  }

  /**
   * Clears the owned reference to NULL.
   */
  inline void clear()
  {
    Py_XDECREF(m_obj);
    m_obj = NULL;
  }

  /** Returns a borrowed reference. */
  inline PyObject *get() const { return m_obj; }

  /** Returns a borrowed reference. */
  inline operator PyObject *() { return m_obj; }

  /**
   * Returns the reference owned by this object,
   * use it like "return obj.release()". After the
   * call, this object contains NULL.
   */
  inline PyObject *release()
  {
    PyObject *result = m_obj;
    m_obj = NULL;
    return result;
  }
};

class PyGILState_RAII {
  PyGILState_STATE m_gstate;

  PyGILState_RAII(const PyGILState_RAII &);
  PyGILState_RAII &operator=(const PyGILState_RAII &);

public:
  inline PyGILState_RAII() { m_gstate = PyGILState_Ensure(); }

  inline ~PyGILState_RAII() { PyGILState_Release(m_gstate); }
};

/**
 * Function which casts the parameter to
 * a PyObject pointer and calls Py_XDECREF on it.
 */
inline void py_decref_function(void *obj)
{
  // Because dynd in general is intended to do things multi-threaded
  // (eventually),
  // the decref function needs to be threadsafe. The way to do that is to ensure
  // we're holding the GIL. This is slower than normal DECREF, but because the
  // reference count isn't an atomic variable, this appears to be the best we
  // can do.
  if (obj != NULL) {
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    Py_DECREF((PyObject *)obj);

    PyGILState_Release(gstate);
  }
}

inline intptr_t pyobject_as_index(PyObject *index)
{
  pyobject_ownref start_obj(PyNumber_Index(index));
  intptr_t result;
  if (PyLong_Check(start_obj.get())) {
    result = PyLong_AsSsize_t(start_obj.get());
#if PY_VERSION_HEX < 0x03000000
  }
  else if (PyInt_Check(start_obj.get())) {
    result = PyInt_AS_LONG(start_obj.get());
#endif
  }
  else {
    throw std::runtime_error(
        "Value returned from PyNumber_Index is not an int or long");
  }
  if (result == -1 && PyErr_Occurred()) {
    throw std::exception();
  }
  return result;
}

inline int pyobject_as_int_index(PyObject *index)
{
  pyobject_ownref start_obj(PyNumber_Index(index));
#if PY_VERSION_HEX >= 0x03000000
  long result = PyLong_AsLong(start_obj);
#else
  long result = PyInt_AsLong(start_obj);
#endif
  if (result == -1 && PyErr_Occurred()) {
    throw std::exception();
  }
  if (((unsigned long)result & 0xffffffffu) != (unsigned long)result) {
    throw std::overflow_error(
        "overflow converting Python integer to 32-bit int");
  }
  return (int)result;
}

inline dynd::irange pyobject_as_irange(PyObject *index)
{
  if (PySlice_Check(index)) {
    dynd::irange result;
    PySliceObject *slice = (PySliceObject *)index;
    if (slice->start != Py_None) {
      result.set_start(pyobject_as_index(slice->start));
    }
    if (slice->stop != Py_None) {
      result.set_finish(pyobject_as_index(slice->stop));
    }
    if (slice->step != Py_None) {
      result.set_step(pyobject_as_index(slice->step));
    }
    return result;
  }
  else {
    return dynd::irange(pyobject_as_index(index));
  }
}

inline std::string pystring_as_string(PyObject *str)
{
  char *data = NULL;
  Py_ssize_t len = 0;
  if (PyUnicode_Check(str)) {
    pyobject_ownref utf8(PyUnicode_AsUTF8String(str));

#if PY_VERSION_HEX >= 0x03000000
    if (PyBytes_AsStringAndSize(utf8.get(), &data, &len) < 0) {
#else
    if (PyString_AsStringAndSize(utf8.get(), &data, &len) < 0) {
#endif
      throw std::runtime_error("Error getting string data");
    }
    return std::string(data, len);
#if PY_VERSION_HEX < 0x03000000
  }
  else if (PyString_Check(str)) {
    if (PyString_AsStringAndSize(str, &data, &len) < 0) {
      throw std::runtime_error("Error getting string data");
    }
    return std::string(data, len);
#endif
  }
  else {
    throw dynd::type_error("Cannot convert pyobject to string");
  }
}

inline PyObject *pystring_from_string(const char *str)
{
#if PY_VERSION_HEX >= 0x03000000
  return PyUnicode_FromString(str);
#else
  return PyString_FromString(str);
#endif
}
inline PyObject *pystring_from_string(const std::string &str)
{
#if PY_VERSION_HEX >= 0x03000000
  return PyUnicode_FromStringAndSize(str.data(), str.size());
#else
  return PyString_FromStringAndSize(str.data(), str.size());
#endif
}
inline std::string pyobject_repr(PyObject *obj)
{
  pyobject_ownref src_repr(PyObject_Repr(obj));
  return pystring_as_string(src_repr.get());
}

inline void pyobject_as_vector_string(PyObject *list_string,
                                      std::vector<std::string> &vector_string)
{
  Py_ssize_t size = PySequence_Size(list_string);
  vector_string.resize(size);
  for (Py_ssize_t i = 0; i < size; ++i) {
    pyobject_ownref item(PySequence_GetItem(list_string, i));
    vector_string[i] = pystring_as_string(item.get());
  }
}

inline void pyobject_as_vector_intp(PyObject *list_index,
                                    std::vector<intptr_t> &vector_intp,
                                    bool allow_int)
{
  if (allow_int) {
    // If permitted, convert an int into a size-1 list
    if (PyLong_Check(list_index)) {
      intptr_t v = PyLong_AsSsize_t(list_index);
      if (v == -1 && PyErr_Occurred()) {
        throw std::runtime_error("error converting int");
      }
      vector_intp.resize(1);
      vector_intp[0] = v;
      return;
    }
#if PY_VERSION_HEX < 0x03000000
    if (PyInt_Check(list_index)) {
      vector_intp.resize(1);
      vector_intp[0] = PyInt_AS_LONG(list_index);
      return;
    }
#endif
    if (PyIndex_Check(list_index)) {
      PyObject *idx_obj = PyNumber_Index(list_index);
      if (idx_obj != NULL) {
        intptr_t v = PyLong_AsSsize_t(idx_obj);
        Py_DECREF(idx_obj);
        if (v == -1 && PyErr_Occurred()) {
          throw std::exception();
        }
        vector_intp.resize(1);
        vector_intp[0] = v;
        return;
      }
      else if (PyErr_ExceptionMatches(PyExc_TypeError)) {
        // Swallow a type error, fall through to the sequence code
        PyErr_Clear();
      }
      else {
        // Propagate the error
        throw std::exception();
      }
    }
  }
  Py_ssize_t size = PySequence_Size(list_index);
  vector_intp.resize(size);
  for (Py_ssize_t i = 0; i < size; ++i) {
    pyobject_ownref item(PySequence_GetItem(list_index, i));
    vector_intp[i] = pyobject_as_index(item.get());
  }
}

/**
 * Same as PySequence_Size, but throws a C++
 * exception on error.
 */
inline Py_ssize_t pysequence_size(PyObject *seq)
{
  Py_ssize_t s = PySequence_Size(seq);
  if (s == -1 && PyErr_Occurred()) {
    throw std::exception();
  }
  return s;
}

/**
 * Same as PyDict_GetItemString, but throws a
 * C++ exception on error.
 */
inline PyObject *pydict_getitemstring(PyObject *dp, const char *key)
{
  PyObject *result = PyDict_GetItemString(dp, key);
  if (result == NULL) {
    throw std::exception();
  }
  return result;
}

inline PyObject *intptr_array_as_tuple(size_t size, const intptr_t *values)
{
  PyObject *result = PyTuple_New(size);
  if (result == NULL) {
    return NULL;
  }

  for (size_t i = 0; i < size; i++) {
    PyObject *o = PyLong_FromLongLong(values[i]);
    if (o == NULL) {
      Py_DECREF(result);
      return NULL;
    }
    PyTuple_SET_ITEM(result, i, o);
  }

  return result;
}

// Helper function for pyarg_axis_argument.
inline void mark_axis(PyObject *int_axis, int ndim, dynd::bool1 *reduce_axes)
{
  pyobject_ownref value_obj(PyNumber_Index(int_axis));
  long value = PyLong_AsLong(value_obj);
  if (value == -1 && PyErr_Occurred()) {
    throw std::runtime_error("error getting integer for axis argument");
  }

  if (value >= ndim || value < -ndim) {
    throw dynd::axis_out_of_bounds(value, ndim);
  }
  else if (value < 0) {
    value += ndim;
  }

  if (!reduce_axes[value]) {
    reduce_axes[value] = true;
  }
  else {
    std::stringstream ss;
    ss << "axis " << value << " is specified more than once";
    throw std::runtime_error(ss.str());
  }
}

/**
 * Parses the axis argument, which may be either a single index
 * or a tuple of indices. They are converted into a boolean array
 * which is set to true whereever a reduction axis is provided.
 *
 * Returns the number of axes which were set.
 */
inline int pyarg_axis_argument(PyObject *axis, int ndim,
                               dynd::bool1 *reduce_axes)
{
  int axis_count = 0;

  if (axis == NULL || axis == Py_None) {
    // None means use all the axes
    for (int i = 0; i < ndim; ++i) {
      reduce_axes[i] = true;
    }
    axis_count = ndim;
  }
  else {
    // Start with no axes marked
    for (int i = 0; i < ndim; ++i) {
      reduce_axes[i] = false;
    }
    if (PyTuple_Check(axis)) {
      // A tuple of axes
      Py_ssize_t size = PyTuple_GET_SIZE(axis);
      for (Py_ssize_t i = 0; i < size; ++i) {
        mark_axis(PyTuple_GET_ITEM(axis, i), ndim, reduce_axes);
        axis_count++;
      }
    }
    else {
      // Just one axis
      mark_axis(axis, ndim, reduce_axes);
      axis_count = 1;
    }
  }

  return axis_count;
}

} // namespace pydynd

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

template <bool not_null = false>
inline void incref(PyObject *) noexcept;

template <>
inline void incref<false>(PyObject *obj) noexcept
{
  Py_XINCREF(obj);
}

template <>
inline void incref<true>(PyObject *obj) noexcept
{
  Py_INCREF(obj);
}

template <bool not_null>
inline void decref(PyObject *) noexcept;

template <>
inline void decref<false>(PyObject *obj) noexcept
{
  Py_XDECREF(obj);
}

template <>
inline void decref<true>(PyObject *obj) noexcept
{
  Py_DECREF(obj);
}

template <bool owns_ref, bool not_null>
inline void incref_if_owned(PyObject *obj) noexcept;

template <>
inline void incref_if_owned<true, true>(PyObject *obj) noexcept
{
  Py_INCREF(obj);
}

template <>
inline void incref_if_owned<true, false>(PyObject *obj) noexcept
{
  Py_XINCREF(obj);
}

template <>
inline void incref_if_owned<false, true>(PyObject *obj) noexcept
{
}

template <>
inline void incref_if_owned<false, false>(PyObject *obj) noexcept
{
}

template <bool owns_ref, bool not_null>
inline void decref_if_owned(PyObject *obj) noexcept;

template <>
inline void decref_if_owned<true, true>(PyObject *obj) noexcept
{
  Py_DECREF(obj);
}

template <>
inline void decref_if_owned<true, false>(PyObject *obj) noexcept
{
  Py_XDECREF(obj);
}

template <>
inline void decref_if_owned<false, true>(PyObject *obj) noexcept
{
}

template <>
inline void decref_if_owned<false, false>(PyObject *obj) noexcept
{
}

template <bool owns_ref = true, bool not_null = true>
class py_ref_tmpl {
  PyObject *o;

public:
  // Explicit specializations for default constructor
  // are defined after the class declaration.
  // If the class is allowed to be null, it default-initializes to null.
  // If the class is not allowed to be null, it default-initializes to None.
  py_ref_tmpl() noexcept;

  // First define an accessor to get the PyObject pointer from
  // the wrapper class.
  PyObject *get() noexcept { return o; }

  // All copy and move constructors are defined as noexcept.
  // If the conversion or move operation would move from a type
  // that can be null to one that can not, the corresponding
  // constructor is declared explicit.

  /* If:
   *    this type allows null,
   * Or:
   *   this type doesn't allow null
   *   and the input type doesn't allow null,
   * Then:
   *    Allow implicit conversions from the other py_ref_tmpl type to this type.
   */
  template <bool other_owns, bool other_not_null,
            typename std::enable_if_t<!not_null || (not_null && other_not_null)> * = nullptr>
  py_ref_tmpl(const py_ref_tmpl<other_owns, other_not_null> &other) noexcept
  {
    o = other.o;
    incref_if_owned<owns_ref, not_null>(o);
  }

  /* If:
   *    this type doesn't allow null,
   *    and the input type does,
   * Then:
   *    Require that conversions from the other py_ref_tmpl type to this type be explcit.
   */
  template <bool other_owns, bool other_not_null, typename std::enable_if_t<not_null && !other_not_null> * = nullptr>
  explicit py_ref_tmpl(const py_ref_tmpl<other_owns, other_not_null> &other) noexcept
  {
    o = other.o;
    incref_if_owned<owns_ref, not_null>(o);
  }

  // Move constructors are really only useful for moving
  // between types that own their references.
  // Only define them for those cases.

  /* If:
   *    both this type and the input type own their reference,
   *    and we are not moving from a type that allows null values to one that does not,
   * Then:
   *    a move operation can be implicitly performed.
   */
  template <bool other_not_null, typename std::enable_if_t<owns_ref && (other_not_null || !not_null)> * = nullptr>
  py_ref_tmpl(py_ref_tmpl<true, other_not_null> &&other) noexcept
  {
    o = other.o;
  }

  /* If:
   *    both this type and the input type own their reference,
   *    and we are moving from a type that allows null values to one that does not,
   * Then:
   *    only an explicit move operation can be performed.
   */
  template <bool other_not_null, typename std::enable_if_t<owns_ref && !other_not_null && not_null> * = nullptr>
  explicit py_ref_tmpl(py_ref_tmpl<true, other_not_null> &&other) noexcept
  {
    o = other.o;
  }

  /* When constructing from a PyObject*, the boolean value `consume_ref` must
   * be passed to the constructor so it is clear whether or not to
   * increment, decrement, or do nothing the reference when storing it in the
   * desired smart pointer type. If you do not intend for the smart
   * pointer to capture the reference that you own, you should
   * specify `consume_ref` as false regardless of whether or not you
   * own the reference represented in the PyObject* you pass in.
   */
  explicit py_ref_tmpl(PyObject *obj, bool consume_ref) noexcept
  {
    o = obj;
    if (consume_ref) {
      decref_if_owned<!owns_ref, not_null>(o);
    }
    else {
      incref_if_owned<owns_ref, not_null>(o);
    }
  }

  ~py_ref_tmpl() { decref_if_owned<owns_ref, not_null>(o); }

  // For assignment operators, only allow assignment
  // in cases where implicit conversions are also allowed.
  // This forces explicit handling of cases that could
  // need to have an exception raised.
  // For that reason, these are all marked as noexcept.
  // Assignment never comsumes a reference.

  /* If:
   *    this type allows null,
   * Or:
   *   this type doesn't allow null
   *   and the input type doesn't allow null,
   * Then:
   *    Allow assignment from the other py_ref_tmpl type to this type.
   */
  template <bool other_owns, bool other_not_null,
            typename std::enable_if_t<!not_null || (not_null && other_not_null)> * = nullptr>
  py_ref_tmpl<owns_ref, not_null> operator=(const py_ref_tmpl<other_owns, other_not_null> &other) noexcept
  {
    decref_if_owned<owns_ref, not_null>(o);
    o = other.o;
    incref_if_owned<owns_ref, not_null>(o);
  }

  // Check if the wrapped pointer is null.
  // Always return true if it is not null by definition.
  bool is_null() noexcept
  {
    // This function will be a no-op returning true
    // when not_null is false.
    if (not_null || (o != nullptr)) {
      return false;
    }
    return true;
  }

  // A debug version of is_null with a purely dynamic check.
  bool is_null_dbg() noexcept { return o != nullptr; }
};

// Default constructors for various cases.
template <>
inline py_ref_tmpl<true, true>::py_ref_tmpl() noexcept
{
  o = Py_None;
  incref<true>(o);
}

template <>
inline py_ref_tmpl<true, false>::py_ref_tmpl() noexcept
{
  o = nullptr;
}

template <>
inline py_ref_tmpl<false, true>::py_ref_tmpl() noexcept
{
  o = Py_None;
}

template <>
inline py_ref_tmpl<false, false>::py_ref_tmpl() noexcept
{
  o = nullptr;
}

// Convenience aliases for the templated smart pointer classes.

using py_ref = py_ref_tmpl<true, true>;

using py_ref_with_null = py_ref_tmpl<true, false>;

using py_borref = py_ref_tmpl<false, true>;

using py_borref_with_null = py_ref_tmpl<false, false>;

// To help with the transition to the new classes.
using pyobject_ownref = py_ref_tmpl<true, false>;

/* Check if a wrapped pointer is null.
 * If it is not, return the pointer
 * wrapped in the corresponding not_null wrapper type.
 * If it is, raise an exception.
 * This can be used to forward exceptions from Python.
 */
template <bool owns_ref, bool not_null>
py_ref_tmpl<owns_ref, true> check_null(py_ref_tmpl<owns_ref, not_null> &o)
{
  if (o.is_null()) {
    throw std::runtime_error("Unexpected null pointer.");
  }
  return reinterpret_cast<py_ref_tmpl<owns_ref, true>>(o);
}

/* Capture a new reference if it is not null.
 * Throw an exception if it is.
 */
inline py_ref capture_if_not_null(PyObject *o)
{
  if (o == nullptr) {
    throw std::runtime_error("Unexpected null pouter.");
  }
  return py_ref(o, true);
}

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
  py_ref start_obj = capture_if_not_null(PyNumber_Index(index));
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
    throw std::runtime_error("Value returned from PyNumber_Index is not an int or long");
  }
  if (result == -1 && PyErr_Occurred()) {
    throw std::exception();
  }
  return result;
}

inline int pyobject_as_int_index(PyObject *index)
{
  py_ref start_obj = capture_if_not_null(PyNumber_Index(index));
#if PY_VERSION_HEX >= 0x03000000
  long result = PyLong_AsLong(start_obj.get());
#else
  long result = PyInt_AsLong(start_obj.get());
#endif
  if (result == -1 && PyErr_Occurred()) {
    throw std::exception();
  }
  if (((unsigned long)result & 0xffffffffu) != (unsigned long)result) {
    throw std::overflow_error("overflow converting Python integer to 32-bit int");
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
    py_ref utf8 = capture_if_not_null(PyUnicode_AsUTF8String(str));

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
  py_ref src_repr = capture_if_not_null(PyObject_Repr(obj));
  return pystring_as_string(src_repr.get());
}

inline void pyobject_as_vector_string(PyObject *list_string, std::vector<std::string> &vector_string)
{
  Py_ssize_t size = PySequence_Size(list_string);
  vector_string.resize(size);
  for (Py_ssize_t i = 0; i < size; ++i) {
    py_ref item = capture_if_not_null(PySequence_GetItem(list_string, i));
    vector_string[i] = pystring_as_string(item.get());
  }
}

inline void pyobject_as_vector_intp(PyObject *list_index, std::vector<intptr_t> &vector_intp, bool allow_int)
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
    py_ref item = capture_if_not_null(PySequence_GetItem(list_index, i));
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
  py_ref value_obj = capture_if_not_null(PyNumber_Index(int_axis));
  long value = PyLong_AsLong(value_obj.get());
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
inline int pyarg_axis_argument(PyObject *axis, int ndim, dynd::bool1 *reduce_axes)
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

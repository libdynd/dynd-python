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

// if if_condition is true, then assert assert_condition.
#define PYDYND_ASSERT_IF(if_condition, assert_condition) (assert((!(if_condition)) || (assert_condition)))

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
  assert(obj != nullptr);
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
  assert(obj != nullptr);
  Py_DECREF(obj);
}

template <bool owns_ref, bool not_null>
inline void incref_if_owned(PyObject *obj) noexcept;

template <>
inline void incref_if_owned<true, true>(PyObject *obj) noexcept
{
  assert(obj != nullptr);
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
  assert(obj != nullptr);
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
  // All versions default-initialize to null.
  // This allows the smart pointer class to be moved from.
  // Whether or not the class is allowed to be null only changes
  // when zerochecks/exceptions may occur and when assertions are used.
  py_ref_tmpl() noexcept : o(nullptr){};

  // First define an accessor to get the PyObject pointer from
  // the wrapper class.
  PyObject *get() noexcept
  {
    PYDYND_ASSERT_IF(not_null, o != nullptr);
    return o;
  }

  // All copy and move constructors are designated implicit.
  // Unless they convert between a null type to a not_null type,
  // they are also declared noexcept.

  /* If:
   *    this type allows null,
   * Or:
   *   this type doesn't allow null
   *   and the input type doesn't allow null,
   * Then:
   *    Conversions from the other py_ref_tmpl type to this type do not raise exceptions.
   */
  template <bool other_owns_ref, bool other_not_null,
            typename std::enable_if_t<(other_not_null && not_null) || !not_null> * = nullptr>
  py_ref_tmpl(const py_ref_tmpl<other_owns_ref, other_not_null> &other) noexcept
  {
    PYDYND_ASSERT_IF(not_null || other_not_null, other.o != nullptr);
    o = other.o;
    incref_if_owned<owns_ref, not_null>(o);
  }

  // Should this one be declared explicit since it can throw?
  /* If:
   *    this type doesn't allow null,
   *    and the input type does,
   * Then:
   *    Allow implicit conversions from the input type to this type, but
   *    raise an exception if the input is null.
   */
  template <bool other_owns_ref, bool other_not_null,
            typename std::enable_if_t<!other_not_null && not_null> * = nullptr>
  py_ref_tmpl(const py_ref_tmpl<other_owns_ref, other_not_null> &other)
  {
    if (other.o != nullptr) {
      o = other.o;
      incref_if_owned<owns_ref, not_null>(o);
    }
    else {
      throw std::invalid_argument("Cannot convert null valued pointer to non-null reference.");
    }
  }

  // Move constructors are really only useful for moving
  // from types that own their references.
  // Only define them for those cases.

  /* If:
   *    the input type owns its reference,
   *    and we are not moving from a type that allows null values to one that does not,
   * Then:
   *    a move operation is defined and will not raise an exception.
   */
  template <bool other_not_null, typename std::enable_if_t<(other_not_null || !not_null)> * = nullptr>
  py_ref_tmpl(py_ref_tmpl<true, other_not_null> &&other) noexcept
  {
    // If this type is a non-null type, the assigned value should not be null.
    // If the other type is a non-null type, the provided value should not be null,
    // unless it is being used uninitialized or after being moved from.
    PYDYND_ASSERT_IF(not_null || other_not_null, other.o != nullptr);
    // Use get to assert that other.o is not null if other_not_null is true.
    o = other.o;
    other.o = nullptr;
    // The other type owns its reference.
    // If this one does not, decref the new pointer.
    // If this type is a non-null type, the assigned value should not be null.
    decref_if_owned<!owns_ref, not_null>(o);
    // Make sure other can be destructed without decrefing
    // this object's wrapped pointer.
  }

  /* If:
   *    the input type owns its reference,
   *    and we are moving from a type that allows null values to one that does not,
   * Then:
   *    a move may be performed, but may also raise an exception.
   */
  template <bool other_not_null, typename std::enable_if_t<!other_not_null && not_null> * = nullptr>
  py_ref_tmpl(py_ref_tmpl<true, other_not_null> &&other)
  {
    if (other.o != nullptr) {
      o = other.o;
      other.o = nullptr;
      // The other type owns its reference.
      // If this one does not, decref the new pointer.
      // The assigned value is already known not be null.
      decref_if_owned<!owns_ref, not_null>(o);
      // Make sure other can be destructed without decrefing
      // this object's wrapped pointer.
    }
    else {
      throw std::invalid_argument("Cannot convert null valued pointer to non-null reference.");
    }
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

  ~py_ref_tmpl()
  {
    // A smart pointer wrapping a PyObject* still needs to be safe to
    // destruct after it has been moved from.
    // Because of that, always zero check when doing the last decref.
    decref_if_owned<owns_ref, false>(o);
  }

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
   *    Assignment from the other py_ref_tmpl type to this type may not raise an exception.
   */
  template <bool other_owns_ref, bool other_not_null,
            typename std::enable_if_t<(other_not_null && not_null) || !not_null> * = nullptr>
  py_ref_tmpl<owns_ref, not_null> &operator=(const py_ref_tmpl<other_owns_ref, other_not_null> &other) noexcept
  {
    PYDYND_ASSERT_IF(not_null || other_not_null, other.o != nullptr);
    // Nullcheck when doing decref in case this object
    // is uninitialized or has been moved from.
    decref_if_owned<owns_ref, false>(o);
    o = other.o;
    incref_if_owned<owns_ref, not_null>(o);
    return *this;
  }

  /* If:
   *    this type does not allow null,
   *    and the other type does,
   * Then:
   *    Assignment from the other py_ref_tmpl type to this type may raise an exception.
   */
  template <bool other_owns_ref, bool other_not_null,
            typename std::enable_if_t<!other_not_null && not_null> * = nullptr>
  py_ref_tmpl<owns_ref, not_null> &operator=(const py_ref_tmpl<other_owns_ref, other_not_null> &other) noexcept
  {
    if (other.o != nullptr) {
      // Nullcheck when doing decref in case this object
      // is uninitialized or has been moved from.
      decref_if_owned<owns_ref, false>(o);
      o = other.o;
      incref_if_owned<owns_ref, not_null>(o);
      return *this;
    }
    else {
      throw std::invalid_argument("Cannot assign null valued pointer to non-null reference.");
    }
  }

  // Same as previous two, except these assign from rvalues rather than lvalues.

  /* If:
   *    this type allows null,
   * Or:
   *   this type doesn't allow null
   *   and the input type doesn't allow null,
   * Then:
   *    Assignment from the other py_ref_tmpl type to this type may not raise an exception.
   */
  template <bool other_owns_ref, bool other_not_null,
            typename std::enable_if_t<(other_not_null && not_null) || !not_null> * = nullptr>
  py_ref_tmpl<owns_ref, not_null> &operator=(py_ref_tmpl<other_owns_ref, other_not_null> &&other) noexcept
  {
    PYDYND_ASSERT_IF(not_null || other_not_null, other.o != nullptr);
    // Nullcheck when doing decref in case this object
    // is uninitialized or has been moved from.
    decref_if_owned<owns_ref, false>(o);
    o = other.o;
    other.o = nullptr;
    incref_if_owned<owns_ref && !other_owns_ref, not_null>(o);
    return *this;
  }

  /* If:
   *    this type does not allow null,
   *    and the other type does,
   * Then:
   *    Assignment from the other py_ref_tmpl type to this type may raise an exception.
   */
  template <bool other_owns_ref, bool other_not_null,
            typename std::enable_if_t<!other_not_null && not_null> * = nullptr>
  py_ref_tmpl<owns_ref, not_null> &operator=(py_ref_tmpl<other_owns_ref, other_not_null> &&other) noexcept
  {
    if (other.o != nullptr) {
      // Nullcheck when doing decref in case this object
      // is uninitialized or has been moved from.
      decref_if_owned<owns_ref, false>(o);
      o = other.o;
      other.o = nullptr;
      incref_if_owned<owns_ref && !other_owns_ref, not_null>(o);
      return *this;
    }
    else {
      throw std::invalid_argument("Cannot assign null valued pointer to non-null reference.");
    }
  }

  // Return an owned reference to the encapsulated PyObject as a raw pointer.
  // Set the encapsulated pointer to NULL.
  PyObject *release() noexcept
  {
    PYDYND_ASSERT_IF(not_null, o != nullptr);
    auto ret = o;
    o = nullptr;
    incref_if_owned<!owns_ref, not_null>(ret);
    return ret;
  }

  // Check if the wrapped pointer is null.
  bool is_null() noexcept { return o != nullptr; }

  py_ref_tmpl<false, not_null> borrow() noexcept { return py_ref<false, not_null>(o, false); }
};

// Convenience aliases for the templated smart pointer classes.

using py_ref = py_ref_tmpl<true, true>;

using py_ref_with_null = py_ref_tmpl<true, false>;

using py_borref = py_ref_tmpl<false, true>;

using py_borref_with_null = py_ref_tmpl<false, false>;

/* Capture a new reference if it is not null.
 * Throw an exception if it is.
 */
inline py_ref capture_if_not_null(PyObject *o)
{
  if (o == nullptr) {
    throw std::runtime_error("Unexpected null pointer.");
  }
  auto p = py_ref(o, true);
  return p;
  // return py_ref(o, true);
}

/* Convert to a non-null reference.
 * If the input type allows nulls, explicitly check for null and raise an exception.
 * If the input type does not allow null, this function is marked as noexcept and
 * only checks for via an assert statement in debug builds.
 */
template <bool owns_ref, bool not_null>
inline py_ref_tmpl<owns_ref, true> nullcheck(py_ref_tmpl<owns_ref, not_null> &&obj) noexcept(not_null)
{
  // Route this through the assignment operator since the semantics are the same.
  py_ref_tmpl<owns_ref, true> out = obj;
  return out;
}

/* Convert to a non-null reference.
 * No nullcheck is used, except for an assertion in debug builds.
 * This should be used when the pointer is already known to not be null.
 */
template <bool owns_ref, bool not_null>
inline py_ref_tmpl<owns_ref, true> disallow_null(py_ref_tmpl<owns_ref, not_null> &&obj) noexcept
{
  assert(obj.get() != nullptr);
  return py_ref_tmpl<owns_ref, true>(obj.release(), owns_ref);
}

// RAII class to acquire GIL.
class with_gil {
  PyGILState_STATE m_gstate;

  with_gil(const with_gil &) = delete;

  with_gil &operator=(const with_gil &) = delete;

public:
  inline with_gil() { m_gstate = PyGILState_Ensure(); }

  inline ~with_gil() { PyGILState_Release(m_gstate); }
};

// RAII class to release GIL.
class with_nogil {
  PyThreadState *m_tstate;

  with_nogil(const with_nogil &) = delete;

  with_nogil &operator=(const with_nogil &) = delete;

public:
  inline with_nogil() { m_tstate = PyEval_SaveThread(); }

  inline ~with_nogil() { PyEval_RestoreThread(m_tstate); }
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

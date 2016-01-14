//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__UTILITY_FUNCTIONS_HPP_
#define _DYND__UTILITY_FUNCTIONS_HPP_

#include <Python.h>

#include <sstream>
#include <stdexcept>
#include <string>

#include <dynd/type.hpp>

#include "visibility.hpp"

namespace pydynd {

/**
 * Function which casts the parameter to
 * a PyObject pointer and calls Py_XDECREF on it.
 */
void py_decref_function(void *obj);

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

size_t pyobject_as_size_t(PyObject *obj);
intptr_t pyobject_as_index(PyObject *index);
int pyobject_as_int_index(PyObject *index);
dynd::irange pyobject_as_irange(PyObject *index);
PYDYND_API std::string pystring_as_string(PyObject *str);
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

void pyobject_as_vector__type(PyObject *list_dtype,
                              std::vector<dynd::ndt::type> &vector_dtype);
void pyobject_as_vector_string(PyObject *list_string,
                               std::vector<std::string> &vector_string);
void pyobject_as_vector_intp(PyObject *list_index,
                             std::vector<intptr_t> &vector_intp,
                             bool allow_int);

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

PYDYND_API PyObject *intptr_array_as_tuple(size_t size, const intptr_t *array);

int pyarg_strings_to_int(PyObject *obj, const char *argname, int default_value,
                         const char *string0, int value0, const char *string1,
                         int value1, const char *string2, int value2,
                         const char *string3, int value3);

/**
 * Accepts "readwrite" and "immutable".
 */
uint32_t pyarg_creation_access_flags(PyObject *obj);

} // namespace pydynd

#endif // _DYND__UTILITY_FUNCTIONS_HPP_

//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <Python.h>

namespace pydynd {

template <typename T>
struct PyWrapper {
  PyObject_HEAD;
  T v;
};

template <typename T>
PyTypeObject *&get_wrapper_type()
{
  static PyTypeObject *type = NULL;
  return type;
}

template <typename T>
void set_wrapper_type(PyObject *obj)
{
  get_wrapper_type<T>() = reinterpret_cast<PyTypeObject *>(obj);
}

template <typename T>
PyObject *wrap(const T &v)
{
  PyTypeObject *type = get_wrapper_type<T>();

  PyWrapper<T> *obj = reinterpret_cast<PyWrapper<T> *>(type->tp_alloc(type, 0));
  if (obj == NULL) {
    throw std::runtime_error("");
  }
  new (&obj->v) T(v);
  return reinterpret_cast<PyObject *>(obj);
}

template <typename T>
struct PyWrapperIter {
  typedef T value_type;

  PyObject *iter;
  PyObject *item;

  PyWrapperIter() = default;

  PyWrapperIter(const PyWrapperIter &other) : iter(other.iter), item(other.item)
  {
    Py_INCREF(iter);
    if (item != NULL) {
      Py_INCREF(item);
    }
  }

  ~PyWrapperIter()
  {
    Py_DECREF(iter);
    if (item != NULL) {
      Py_DECREF(item);
    }
  }

  PyWrapperIter &operator=(const PyWrapperIter &other)
  {
    iter = other.iter;
    Py_INCREF(iter);

    item = other.item;
    if (item != NULL) {
      Py_INCREF(item);
    }
  }

  PyWrapperIter &operator++()
  {
    Py_DECREF(item);
    item = PyIter_Next(iter);

    return *this;
  }

  PyWrapperIter operator++(int)
  {
    PyWrapperIter tmp(*this);
    operator++();
    return tmp;
  }

  T &operator*() { return reinterpret_cast<PyWrapper<T> *>(item)->v; }

  bool operator==(const PyWrapperIter &other) const
  {
    return item == other.item;
  }

  bool operator!=(const PyWrapperIter &other) const
  {
    return item != other.item;
  }
};

} // namespace pydynd

namespace std {

template <typename T>
pydynd::PyWrapperIter<T> begin(PyObject *obj)
{
  PyObject *iter = PyObject_GetIter(obj);
  if (iter == NULL) {
    std::cout << "not an iterator" << std::endl;
  }

  pydynd::PyWrapperIter<T> it;
  it.iter = iter;
  it.item = PyIter_Next(it.iter);

  return it;
}

template <typename T>
pydynd::PyWrapperIter<T> end(PyObject *obj)
{
  PyObject *iter = PyObject_GetIter(obj);
  if (iter == NULL) {
    std::cout << "not an iterator" << std::endl;
  }

  pydynd::PyWrapperIter<T> it;
  it.iter = iter;
  it.item = NULL;

  return it;
}

} // namespace std
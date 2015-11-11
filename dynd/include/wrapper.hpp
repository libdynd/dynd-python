//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <Python.h>

#include <dynd/array.hpp>
#include <dynd/type.hpp>
#include <dynd/func/callable.hpp>

#include "config.hpp"

template <typename T>
struct DyND_PyWrapperObject {
  PyObject_HEAD;
  T v;
};

template <typename T>
PyTypeObject *&DyND_PyWrapper_Type();

//template<> PYDYND_API PyTypeObject *&DyND_PyWrapper_Type<dynd::ndt::type>();
//template<> PYDYND_API PyTypeObject *&DyND_PyWrapper_Type<dynd::nd::array>();
//template<> PYDYND_API PyTypeObject *&DyND_PyWrapper_Type<dynd::nd::callable>();

template <typename T>
inline void DyND_PyWrapper_Type(PyObject *obj)
{
  DyND_PyWrapper_Type<T>() = reinterpret_cast<PyTypeObject *>(obj);
}

template <typename T>
PyObject *DyND_PyWrapper_New(const T &v)
{
  PyTypeObject *type = DyND_PyWrapper_Type<T>();

  DyND_PyWrapperObject<T> *obj =
      reinterpret_cast<DyND_PyWrapperObject<T> *>(type->tp_alloc(type, 0));
  if (obj == NULL) {
    throw std::runtime_error("");
  }
  new (&obj->v) T(v);
  return reinterpret_cast<PyObject *>(obj);
}

template <typename T>
int DyND_PyWrapper_Check(PyObject *obj)
{
  return PyObject_TypeCheck(obj, DyND_PyWrapper_Type<T>());
}

template <typename T>
int DyND_PyWrapper_CheckExact(PyObject *obj)
{
  return Py_TYPE(obj) == DyND_PyWrapper_Type<T>();
}

template <typename T>
struct DyND_PyWrapperIter {
  typedef T value_type;

  PyObject *iter;
  PyObject *item;

  DyND_PyWrapperIter() = default;

  DyND_PyWrapperIter(const DyND_PyWrapperIter &other)
      : iter(other.iter), item(other.item)
  {
    Py_INCREF(iter);
    if (item != NULL) {
      Py_INCREF(item);
    }
  }

  ~DyND_PyWrapperIter()
  {
    Py_DECREF(iter);
    if (item != NULL) {
      Py_DECREF(item);
    }
  }

  DyND_PyWrapperIter &operator=(const DyND_PyWrapperIter &other)
  {
    iter = other.iter;
    Py_INCREF(iter);

    item = other.item;
    if (item != NULL) {
      Py_INCREF(item);
    }
  }

  DyND_PyWrapperIter &operator++()
  {
    Py_DECREF(item);
    item = PyIter_Next(iter);

    return *this;
  }

  DyND_PyWrapperIter operator++(int)
  {
    DyND_PyWrapperIter tmp(*this);
    operator++();
    return tmp;
  }

  T &operator*()
  {
    return reinterpret_cast<DyND_PyWrapperObject<T> *>(item)->v;
  }

  bool operator==(const DyND_PyWrapperIter &other) const
  {
    return item == other.item;
  }

  bool operator!=(const DyND_PyWrapperIter &other) const
  {
    return item != other.item;
  }
};

namespace std {

template <typename T>
DyND_PyWrapperIter<T> begin(PyObject *obj)
{
  PyObject *iter = PyObject_GetIter(obj);
  if (iter == NULL) {
    std::cout << "not an iterator" << std::endl;
  }

  DyND_PyWrapperIter<T> it;
  it.iter = iter;
  it.item = PyIter_Next(it.iter);

  return it;
}

template <typename T>
DyND_PyWrapperIter<T> end(PyObject *obj)
{
  PyObject *iter = PyObject_GetIter(obj);
  if (iter == NULL) {
    std::cout << "not an iterator" << std::endl;
  }

  DyND_PyWrapperIter<T> it;
  it.iter = iter;
  it.item = NULL;

  return it;
}

} // namespace std

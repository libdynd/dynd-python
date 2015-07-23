//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <Python.h>

#include "config.hpp"

template <typename T>
struct DyND_PyWrapperObject {
  PyObject_HEAD;
  T v;
};

template <typename T>
PyTypeObject *&DyND_PyWrapper_Type()
{
  static PyTypeObject *type = NULL;
  return type;
}

template <typename T>
void DyND_PyWrapper_Type(PyObject *obj)
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
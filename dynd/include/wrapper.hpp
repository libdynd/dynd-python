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
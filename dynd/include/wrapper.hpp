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
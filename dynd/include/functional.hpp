//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>

#include <Python.h>

#include "visibility.hpp"

PYDYND_API dynd::nd::callable apply(const dynd::ndt::type &tp, PyObject *func);

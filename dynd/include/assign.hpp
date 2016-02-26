//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <Python.h>

#include <dynd/callable.hpp>

#include "numpy_interop.hpp"
#include "visibility.hpp"

PYDYND_API void assign_init();

#if DYND_NUMPY_INTEROP

extern struct assign_to_pyarrayobject
    : dynd::nd::declfunc<assign_to_pyarrayobject> {
  static dynd::nd::callable make();
  static dynd::nd::callable &get();
} assign_to_pyarrayobject;

void array_copy_to_numpy(PyArrayObject *dst_arr, const dynd::ndt::type &src_tp,
                         const char *src_arrmeta, const char *src_data);

#endif // DYND_NUMPY_INTEROP

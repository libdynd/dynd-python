//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include "numpy_interop.hpp"

#include <dynd/func/arrfunc.hpp>

namespace pydynd {

/**
 * This is the arrmeta to provide for the destination
 * void type when instantiating the copy_to_numpy arrfunc.
 */
struct copy_to_numpy_arrmeta {
  // This is the destination PyArray_Descr *.
  PyArray_Descr *dst_dtype;
  // This is the | together of the root data
  // pointer and all the strides/offsets, and
  // can be used to determine the minimum data alignment.
  uintptr_t dst_alignment;
};

extern struct copy_to_numpy : dynd::nd::declfunc<copy_to_numpy> {
  static dynd::nd::arrfunc make();
} copy_to_numpy;

void array_copy_to_numpy(PyArrayObject *dst_arr, const dynd::ndt::type &src_tp,
                         const char *src_arrmeta, const char *src_data,
                         const dynd::eval::eval_context *ectx);

} // namespace pydynd

//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include "numpy_interop.hpp"

#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/kernels/base_virtual_kernel.hpp>
#include <dynd/func/callable.hpp>
#include <dynd/eval/eval_context.hpp>

#include "config.hpp"

namespace pydynd {
namespace nd {

#ifdef DYND_NUMPY_INTEROP

  /**
   * This is the arrmeta to provide for the source
   * void type when instantiating the copy_from_numpy callable.
   */
  struct copy_from_numpy_arrmeta {
    // This is the source PyArray_Descr *.
    PyArray_Descr *src_dtype;
    // This is the | together of the root data
    // pointer and all the strides/offsets, and
    // can be used to determine the minimum data alignment.
    uintptr_t src_alignment;
  };

  extern struct copy_from_numpy : dynd::nd::declfunc<copy_from_numpy> {
    static dynd::nd::callable make();
  } copy_from_numpy;

  void array_copy_from_numpy(const dynd::ndt::type &dst_tp,
                             const char *dst_arrmeta, char *dst_data,
                             PyArrayObject *src_arr,
                             const dynd::eval::eval_context *ectx);

#endif

} // namespace pydynd::nd
} // namespace pydynd

//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include "numpy_interop.hpp"

#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/kernels/virtual.hpp>
#include <dynd/func/arrfunc.hpp>
#include <dynd/eval/eval_context.hpp>

namespace pydynd {

#ifdef DYND_NUMPY_INTEROP

/**
 * This is the arrmeta to provide for the source
 * void type when instantiating the copy_from_numpy arrfunc.
 */
struct copy_from_numpy_arrmeta {
  // This is the source PyArray_Descr *.
  PyArray_Descr *src_dtype;
  // This is the | together of the root data
  // pointer and all the strides/offsets, and
  // can be used to determine the minimum data alignment.
  uintptr_t src_alignment;
};

struct copy_from_numpy_ck : dynd::nd::virtual_ck<copy_from_numpy_ck> {
  static intptr_t instantiate(
      const dynd::arrfunc_type_data *self_af, const dynd::arrfunc_type *af_tp,
      char *data, void *ckb, intptr_t ckb_offset, const dynd::ndt::type &dst_tp,
      const char *dst_arrmeta, intptr_t nsrc, const dynd::ndt::type *src_tp,
      const char *const *src_arrmeta, dynd::kernel_request_t kernreq,
      const dynd::eval::eval_context *ectx, const dynd::nd::array &kwds,
      const std::map<dynd::nd::string, dynd::ndt::type> &tp_vars);
};

extern struct copy_from_numpy : dynd::nd::declfunc<copy_from_numpy> {
  static dynd::nd::arrfunc make();
} copy_from_numpy;

void array_copy_from_numpy(const dynd::ndt::type &dst_tp,
                           const char *dst_arrmeta, char *dst_data,
                           PyArrayObject *src_arr,
                           const dynd::eval::eval_context *ectx);

#endif

} // namespace pydynd

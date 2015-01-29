//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef PYDYND_COPY_FROM_NUMPY_ARRFUNC_HPP
#define PYDYND_COPY_FROM_NUMPY_ARRFUNC_HPP

#include "numpy_interop.hpp"

#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/func/arrfunc.hpp>
#include <dynd/eval/eval_context.hpp>

namespace pydynd {

#ifdef DYND_NUMPY_INTEROP


/**
 * This is the arrmeta to provide for the source
 * void type when instantiating the copy_from_numpy arrfunc.
 */
struct copy_from_numpy_arrmeta {
  // This is either the source PyArrayObject *,
  // or the source PyArray_Descr *.
  PyObject *src_obj;
  // This is the | together of the root data
  // pointer and all the strides/offsets, and
  // can be used to determine the minimum data alignment.
  uintptr_t src_alignment;
};

extern dynd::nd::pod_arrfunc copy_from_numpy;

void array_copy_from_numpy(const dynd::ndt::type &dst_tp,
                           const char *dst_arrmeta, char *dst_data,
                           PyArrayObject *value,
                           const dynd::eval::eval_context *ectx);

void init_copy_from_numpy();
void cleanup_copy_from_numpy();

#endif

} // namespace pydynd

#endif // PYDYND_COPY_FROM_NUMPY_ARRFUNC_HPP

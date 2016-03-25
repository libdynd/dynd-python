//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

#include "types/pyobject_type.hpp"

/**
 * This is the arrmeta to provide for the destination
 * void type when instantiating the copy_to_numpy callable.
 */
struct copy_to_numpy_arrmeta {
  // This is the destination PyArray_Descr *.
  PyArray_Descr *dst_dtype;
  // This is the | together of the root data
  // pointer and all the strides/offsets, and
  // can be used to determine the minimum data alignment.
  uintptr_t dst_alignment;
};

struct strided_of_numpy_arrmeta {
  dynd::fixed_dim_type_arrmeta sdt[NPY_MAXDIMS];
  copy_to_numpy_arrmeta am;
};

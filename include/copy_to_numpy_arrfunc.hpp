//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef PYDYND_COPY_TO_NUMPY_ARRFUNC_HPP
#define PYDYND_COPY_TO_NUMPY_ARRFUNC_HPP

#include <Python.h>

#include <dynd/func/arrfunc.hpp>

namespace pydynd {

/**
 * This is the arrmeta to provide for the destination
 * void type when instantiating the copy_to_numpy arrfunc.
 */
struct copy_to_numpy_arrmeta {
  // This is either the destination PyArrayObject *,
  // or the destination PyArray_Descr *.
  PyObject *dst_obj;
  // This is the | together of the root data
  // pointer and all the strides/offsets, and
  // can be used to determine the minimum data alignment.
  uintptr_t dst_alignment;
};

extern dynd::nd::pod_arrfunc copy_to_numpy;

void init_copy_to_numpy();
void cleanup_copy_to_numpy();

} // namespace pydynd

#endif // PYDYND_COPY_TO_NUMPY_ARRFUNC_HPP

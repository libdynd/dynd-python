//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "numpy_interop.hpp"

#if DYND_NUMPY_INTEROP

#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>

#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/make_lifted_ckernel.hpp>

#include "array_assign_to_numpy.hpp"
#include "copy_to_pyobject_ckernel.hpp"

using namespace std;
using namespace dynd;
using namespace pydynd;

namespace {

struct strided_of_numpy_arrmeta {
  strided_dim_type_arrmeta sdt[NPY_MAXDIMS];
  // This is either the destination PyArrayObject *,
  // or the destination PyArray_Descr *.
  PyObject *dst_obj;
  // This is the | together of the root data
  // pointer and all the strides/offsets, and
  // can be used to determine the minimum data alignment.
  uintptr_t dst_alignment;
};
} // anonymous namespace

/**
 * This sets up a ckernel to copy from a dynd array
 * to a numpy array. The destination numpy array is
 * represented by dst_tp being ``void`` and the dst_arrmeta
 * being a pointer to the ``PyArrayObject *`` for the destination.
 */
static intptr_t instantiate_copy_to_numpy(
    const arrfunc_type_data *self_af, dynd::ckernel_builder *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx)
{
  if (dst_tp.get_type_id() != void_type_id) {
    stringstream ss;
    ss << "Cannot instantiate arrfunc with signature ";
    ss << self_af->func_proto << " with types (";
    ss << src_tp[0] << ") -> " << dst_tp;
    throw type_error(ss.str());
  }

  PyObject *dst_obj = *reinterpret_cast<PyObject *const *>(dst_arrmeta);
  uintptr_t dst_alignment = reinterpret_cast<const uintptr_t *>(dst_arrmeta)[1];
  if (PyArray_Check(dst_obj)) {
    PyArrayObject *dst_arr = reinterpret_cast<PyArrayObject *>(dst_obj);
    intptr_t dst_ndim = PyArray_NDIM(dst_arr);
    intptr_t src_ndim = src_tp[0].get_ndim();

    strided_of_numpy_arrmeta dst_am_holder;
    const char *dst_am = reinterpret_cast<const char *>(
        &dst_am_holder.sdt[NPY_MAXDIMS - dst_ndim]);
    // Fill in metadata for a multi-dim strided array, corresponding
    // to the numpy array, with a void type at the end for the numpy
    // specific data.
    for (intptr_t i = 0; i < dst_ndim; ++i) {
      strided_dim_type_arrmeta &am = dst_am_holder.sdt[NPY_MAXDIMS - dst_ndim + i];
      am.stride = PyArray_STRIDE(dst_arr, (int)i);
      dst_alignment |= static_cast<uintptr_t>(am.stride);
      am.size = PyArray_DIM(dst_arr, (int)i);
    }
    ndt::type dst_am_tp =
        ndt::make_strided_dim(ndt::make_type<void>(), dst_ndim);
    dst_am_holder.dst_obj =
        reinterpret_cast<PyObject *>(PyArray_DTYPE(dst_arr));
    dst_am_holder.dst_alignment = dst_alignment;
    // Use the lifting ckernel mechanism to deal with all the dimensions,
    // calling back to this arrfunc when the dtype is reached
    return make_lifted_expr_ckernel(self_af, ckb, ckb_offset, dst_ndim,
                                    dst_am_tp, dst_am, &src_ndim, src_tp,
                                    src_arrmeta, kernreq, ectx);
  } else {
    PyArray_Descr *dtype = reinterpret_cast<PyArray_Descr *>(dst_obj);
    if (!PyDataType_FLAGCHK(dtype, NPY_ITEM_HASOBJECT)) {
      // If there is no object type in the numpy type, get the dynd equivalent
      // type and use it to do the copying
      ndt::type dst_view_tp = ndt_type_from_numpy_dtype(dtype, dst_alignment);
      return make_assignment_kernel(ckb, ckb_offset, dst_view_tp, NULL, src_tp[0],
                                    src_arrmeta[0], kernreq, ectx);
    } else if (PyDataType_ISOBJECT(dtype)) {
      return make_copy_to_pyobject_kernel(ckb, ckb_offset, src_tp[0],
                                          src_arrmeta[0], true, kernreq, ectx);
    } else {
    }
  }

  return ckb_offset;
}

static nd::arrfunc make_copy_to_numpy_arrfunc()
{
  nd::array out_af = nd::empty(ndt::make_arrfunc());
  arrfunc_type_data *af =
      reinterpret_cast<arrfunc_type_data *>(out_af.get_readwrite_originptr());
  af->func_proto = ndt::type("(A... * T) -> void");
  af->instantiate = &instantiate_copy_to_numpy;
  out_af.flag_as_immutable();
  return out_af;
}

nd::arrfunc pydynd::copy_to_numpy = make_copy_to_numpy_arrfunc();

#endif // DYND_NUMPY_INTEROP

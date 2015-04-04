//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>
#include <datetime.h>

#include "copy_from_numpy_arrfunc.hpp"
#include "copy_from_pyobject_arrfunc.hpp"
#include "numpy_interop.hpp"
#include "utility_functions.hpp"
#include "type_functions.hpp"
#include "array_functions.hpp"
#include "array_from_py_typededuction.hpp"

#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/func/elwise.hpp>
#include <dynd/kernels/tuple_assignment_kernels.hpp>
#include <dynd/types/base_struct_type.hpp>

using namespace std;
using namespace dynd;
using namespace pydynd;

#ifdef DYND_NUMPY_INTEROP

namespace {

struct strided_of_numpy_arrmeta {
  fixed_dim_type_arrmeta sdt[NPY_MAXDIMS];
  copy_from_numpy_arrmeta am;
};

} // anonymous namespace

static intptr_t instantiate_copy_from_numpy(
    const arrfunc_type_data *self_af, const arrfunc_type *af_tp, char *DYND_UNUSED(data), void *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx,
    const nd::array &kwds, const std::map<nd::string, ndt::type> &tp_vars)
{
  if (src_tp[0].get_type_id() != void_type_id) {
    stringstream ss;
    ss << "Cannot instantiate arrfunc copy_from_numpy with signature ";
    ss << af_tp << " with types (";
    ss << src_tp[0] << ") -> " << dst_tp;
    throw type_error(ss.str());
  }

  PyArray_Descr *dtype =
      *reinterpret_cast<PyArray_Descr *const *>(src_arrmeta[0]);
  uintptr_t src_alignment =
      reinterpret_cast<const uintptr_t *>(src_arrmeta[0])[1];

  if (!PyDataType_FLAGCHK(dtype, NPY_ITEM_HASOBJECT)) {
    // If there is no object type in the numpy type, get the dynd equivalent
    // type and use it to do the copying
    ndt::type src_view_tp = ndt_type_from_numpy_dtype(dtype, src_alignment);
    return make_assignment_kernel(NULL, NULL, ckb, ckb_offset, dst_tp,
                                  dst_arrmeta, src_view_tp, NULL, kernreq, ectx,
                                  nd::array());
  }
  else if (PyDataType_ISOBJECT(dtype)) {
    const arrfunc_type_data *af =
        static_cast<dynd::nd::arrfunc>(copy_from_pyobject).get();
    return af->instantiate(
        af, static_cast<dynd::nd::arrfunc>(copy_from_pyobject).get_type(), NULL,
        ckb, ckb_offset, dst_tp, dst_arrmeta, 1, src_tp, src_arrmeta, kernreq,
        ectx, nd::array(), tp_vars);
  }
  else if (PyDataType_HASFIELDS(dtype)) {
    if (dst_tp.get_kind() != struct_kind && dst_tp.get_kind() != tuple_kind) {
      stringstream ss;
      ss << "Cannot assign from numpy type " << pyobject_repr((PyObject *)dtype)
         << " to dynd type " << dst_tp;
      throw invalid_argument(ss.str());
    }

    // Get the fields out of the numpy dtype
    vector<PyArray_Descr *> field_dtypes_orig;
    vector<string> field_names_orig;
    vector<size_t> field_offsets_orig;
    extract_fields_from_numpy_struct(dtype, field_dtypes_orig, field_names_orig,
                                     field_offsets_orig);
    intptr_t field_count = field_dtypes_orig.size();
    if (field_count != dst_tp.extended<base_tuple_type>()->get_field_count()) {
      stringstream ss;
      ss << "Cannot assign from numpy type " << pyobject_repr((PyObject *)dtype)
         << " to dynd type " << dst_tp;
      throw invalid_argument(ss.str());
    }

    // Permute the numpy fields to match with the dynd fields
    vector<PyArray_Descr *> field_dtypes;
    vector<size_t> field_offsets;
    if (dst_tp.get_kind() == struct_kind) {
      field_dtypes.resize(field_count);
      field_offsets.resize(field_count);
      for (intptr_t i = 0; i < field_count; ++i) {
        intptr_t src_i = dst_tp.extended<base_struct_type>()->get_field_index(
            field_names_orig[i]);
        if (src_i >= 0) {
          field_dtypes[src_i] = field_dtypes_orig[i];
          field_offsets[src_i] = field_offsets_orig[i];
        }
        else {
          stringstream ss;
          ss << "Cannot assign from numpy type "
             << pyobject_repr((PyObject *)dtype) << " to dynd type " << dst_tp;
          throw invalid_argument(ss.str());
        }
      }
    }
    else {
      // In the tuple case, use position instead of name
      field_dtypes.swap(field_dtypes_orig);
      field_offsets.swap(field_offsets_orig);
    }

    vector<ndt::type> src_fields_tp(field_count, ndt::make_type<void>());
    vector<copy_from_numpy_arrmeta> src_arrmeta_values(field_count);
    vector<const char *> src_fields_arrmeta(field_count);
    for (intptr_t i = 0; i < field_count; ++i) {
      src_arrmeta_values[i].src_dtype = field_dtypes[i];
      src_arrmeta_values[i].src_alignment = src_alignment | field_offsets[i];
      src_fields_arrmeta[i] =
          reinterpret_cast<const char *>(&src_arrmeta_values[i]);
    }

    const uintptr_t *dst_arrmeta_offsets =
        dst_tp.extended<base_tuple_type>()->get_arrmeta_offsets_raw();
    shortvector<const char *> dst_fields_arrmeta(field_count);
    for (intptr_t i = 0; i != field_count; ++i) {
      dst_fields_arrmeta[i] = dst_arrmeta + dst_arrmeta_offsets[i];
    }

    return make_tuple_unary_op_ckernel(
        self_af, af_tp, ckb, ckb_offset, field_count,
        dst_tp.extended<base_tuple_type>()->get_data_offsets(dst_arrmeta),
        dst_tp.extended<base_tuple_type>()->get_field_types_raw(),
        dst_fields_arrmeta.get(), &field_offsets[0], &src_fields_tp[0],
        &src_fields_arrmeta[0], kernreq, ectx);
  }
  else {
    stringstream ss;
    ss << "TODO: implement assign from numpy type "
       << pyobject_repr((PyObject *)dtype) << " to dynd type " << dst_tp;
    throw invalid_argument(ss.str());
  }
}

static nd::arrfunc make_copy_from_numpy_arrfunc()
{
  nd::array out_af = nd::empty(ndt::type("(void) -> T"));
  arrfunc_type_data *af =
      reinterpret_cast<arrfunc_type_data *>(out_af.get_readwrite_originptr());
  af->instantiate = &instantiate_copy_from_numpy;
  out_af.flag_as_immutable();
  return out_af;
}

dynd::nd::arrfunc pydynd::copy_from_numpy::make() {
  return make_copy_from_numpy_arrfunc();
}

struct pydynd::copy_from_numpy pydynd::copy_from_numpy;
nd::arrfunc elwise_copy_from_numpy =
    nd::functional::elwise(pydynd::copy_from_numpy);

void pydynd::array_copy_from_numpy(const ndt::type &dst_tp,
                                   const char *dst_arrmeta, char *dst_data,
                                   PyArrayObject *src_arr,
                                   const dynd::eval::eval_context *ectx)
{
  intptr_t src_ndim = PyArray_NDIM(src_arr);

  strided_of_numpy_arrmeta src_am_holder;
  const char *src_am = reinterpret_cast<const char *>(
    &src_am_holder.sdt[NPY_MAXDIMS - src_ndim]);
  // Fill in metadata for a multi-dim strided array, corresponding
  // to the numpy array, with a void type at the end for the numpy
  // specific data.
  uintptr_t src_alignment = reinterpret_cast<uintptr_t>(PyArray_DATA(src_arr));
  for (intptr_t i = 0; i < src_ndim; ++i) {
    fixed_dim_type_arrmeta &am = src_am_holder.sdt[NPY_MAXDIMS - src_ndim + i];
    am.dim_size = PyArray_DIM(src_arr, (int)i);
    am.stride = am.dim_size != 1 ? PyArray_STRIDE(src_arr, (int)i) : 0;
    src_alignment |= static_cast<uintptr_t>(am.stride);
  }
  ndt::type src_tp =
      ndt::make_type(src_ndim, PyArray_SHAPE(src_arr), ndt::make_type<void>());
  src_am_holder.am.src_dtype = PyArray_DTYPE(src_arr);
  src_am_holder.am.src_alignment = src_alignment;

  const arrfunc_type_data *af =
      static_cast<dynd::nd::arrfunc>(elwise_copy_from_numpy).get();
  unary_ckernel_builder ckb;
  af->instantiate(
      af, static_cast<dynd::nd::arrfunc>(elwise_copy_from_numpy).get_type(),
      NULL, &ckb, 0, dst_tp, dst_arrmeta, 1, &src_tp, &src_am,
      kernel_request_single, ectx, nd::array(),
      std::map<nd::string, ndt::type>());
  ckb(dst_data, (char *)PyArray_DATA(src_arr));
}

#endif // DYND_NUMPY_INTEROP

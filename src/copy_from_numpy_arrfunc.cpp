//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
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
    const arrfunc_type_data *self_af, const arrfunc_type *af_tp, void *ckb,
    intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
    const ndt::type *src_tp, const char *const *src_arrmeta,
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

  if (!kwds.is_null()) {
    throw invalid_argument("unexpected non-NULL kwds value to "
                           "copy_from_numpy instantiation");
  }

  PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src_arrmeta[0]);
  uintptr_t src_alignment =
      reinterpret_cast<const uintptr_t *>(src_arrmeta[0])[1];

  if (PyArray_Check(src_obj)) {
    PyArrayObject *src_arr = reinterpret_cast<PyArrayObject *>(src_obj);
    intptr_t src_ndim = PyArray_NDIM(src_arr);
    src_alignment |= reinterpret_cast<uintptr_t>(PyArray_DATA(src_arr));

    strided_of_numpy_arrmeta src_am_holder;
    const char *src_am = reinterpret_cast<const char *>(
        &src_am_holder.sdt[NPY_MAXDIMS - src_ndim]);
    // Fill in metadata for a multi-dim strided array, corresponding
    // to the numpy array, with a void type at the end for the numpy
    // specific data.
    for (intptr_t i = 0; i < src_ndim; ++i) {
      fixed_dim_type_arrmeta &am =
          src_am_holder.sdt[NPY_MAXDIMS - src_ndim + i];
      am.dim_size = PyArray_DIM(src_arr, (int)i);
      am.stride = am.dim_size != 1 ? PyArray_STRIDE(src_arr, (int)i) : 0;
      src_alignment |= static_cast<uintptr_t>(am.stride);
    }
    ndt::type src_am_tp = ndt::make_type(src_ndim, PyArray_SHAPE(src_arr),
                                         ndt::make_type<void>());
    src_am_holder.am.src_obj =
        reinterpret_cast<PyObject *>(PyArray_DTYPE(src_arr));
    src_am_holder.am.src_alignment = src_alignment;
    // Use the lifting ckernel mechanism to deal with all the dimensions,
    // calling back to this arrfunc when the dtype is reached
    return nd::elwise.instantiate(self_af, af_tp, ckb, ckb_offset, dst_tp,
                                  dst_arrmeta, &src_am_tp, &src_am, kernreq,
                                  ectx, nd::array(), tp_vars);
  } else {
    PyArray_Descr *dtype = reinterpret_cast<PyArray_Descr *>(src_obj);
    if (!PyDataType_FLAGCHK(dtype, NPY_ITEM_HASOBJECT)) {
      // If there is no object type in the numpy type, get the dynd equivalent
      // type and use it to do the copying
      ndt::type src_view_tp = ndt_type_from_numpy_dtype(dtype, src_alignment);
      return make_assignment_kernel(NULL, NULL, ckb, ckb_offset, dst_tp,
                                    dst_arrmeta, src_view_tp, NULL, kernreq,
                                    ectx, nd::array());
    } else if (PyDataType_ISOBJECT(dtype)) {
      const arrfunc_type_data *af = copy_from_pyobject.get();
      return af->instantiate(af, copy_from_pyobject.get_type(), ckb, ckb_offset,
                             dst_tp, dst_arrmeta, src_tp, src_arrmeta, kernreq,
                             ectx, nd::array(), tp_vars);
    } else if (PyDataType_HASFIELDS(dtype)) {
      if (dst_tp.get_kind() != struct_kind && dst_tp.get_kind() != tuple_kind) {
        stringstream ss;
        ss << "Cannot assign from numpy type "
           << pyobject_repr((PyObject *)dtype) << " to dynd type " << dst_tp;
        throw invalid_argument(ss.str());
      }

      // Get the fields out of the numpy dtype
      vector<PyArray_Descr *> field_dtypes_orig;
      vector<string> field_names_orig;
      vector<size_t> field_offsets_orig;
      extract_fields_from_numpy_struct(dtype, field_dtypes_orig,
                                       field_names_orig, field_offsets_orig);
      intptr_t field_count = field_dtypes_orig.size();
      if (field_count !=
          dst_tp.extended<base_tuple_type>()->get_field_count()) {
        stringstream ss;
        ss << "Cannot assign from numpy type "
           << pyobject_repr((PyObject *)dtype) << " to dynd type " << dst_tp;
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
          } else {
            stringstream ss;
            ss << "Cannot assign from numpy type "
               << pyobject_repr((PyObject *)dtype) << " to dynd type "
               << dst_tp;
            throw invalid_argument(ss.str());
          }
        }
      } else {
        // In the tuple case, use position instead of name
        field_dtypes.swap(field_dtypes_orig);
        field_offsets.swap(field_offsets_orig);
      }

      vector<ndt::type> src_fields_tp(field_count, ndt::make_type<void>());
      vector<copy_from_numpy_arrmeta> src_arrmeta_values(field_count);
      vector<const char *> src_fields_arrmeta(field_count);
      for (intptr_t i = 0; i < field_count; ++i) {
        src_arrmeta_values[i].src_obj = (PyObject *)field_dtypes[i];
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
    } else {
      stringstream ss;
      ss << "TODO: implement assign from numpy type "
         << pyobject_repr((PyObject *)dtype) << " to dynd type " << dst_tp;
      throw invalid_argument(ss.str());
    }
  }
}

static nd::arrfunc make_copy_from_numpy_arrfunc()
{
  nd::array out_af = nd::empty(ndt::type("(void) -> A... * T"));
  arrfunc_type_data *af =
      reinterpret_cast<arrfunc_type_data *>(out_af.get_readwrite_originptr());
  af->instantiate = &instantiate_copy_from_numpy;
  out_af.flag_as_immutable();
  return out_af;
}

dynd::nd::pod_arrfunc pydynd::copy_from_numpy;

void pydynd::init_copy_from_numpy()
{
  pydynd::copy_from_numpy.init(make_copy_from_numpy_arrfunc());
}

void pydynd::cleanup_copy_from_numpy() { pydynd::copy_from_numpy.cleanup(); }

void pydynd::array_copy_from_numpy(const ndt::type &dst_tp,
                                   const char *dst_arrmeta, char *dst_data,
                                   PyArrayObject *value,
                                   const dynd::eval::eval_context *ectx)
{
  unary_ckernel_builder ckb;
  copy_from_numpy_arrmeta src_arrmeta;
  src_arrmeta.src_obj = (PyObject *)value;
  src_arrmeta.src_alignment = 0;
  const char *src_arrmeta_ptr = reinterpret_cast<const char *>(&src_arrmeta);
  const arrfunc_type_data *af = copy_from_numpy.get();
  ndt::type src_tp = ndt::make_type<void>();
  af->instantiate(af, copy_from_numpy.get_type(), &ckb, 0, dst_tp, dst_arrmeta,
                  &src_tp, &src_arrmeta_ptr, kernel_request_single,
                  &eval::default_eval_context, nd::array(), std::map<nd::string, ndt::type>());
  ckb(dst_data, (char *)PyArray_DATA(value));
}

#endif // DYND_NUMPY_INTEROP

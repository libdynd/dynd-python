//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>
#include <datetime.h>

#include "array_functions.hpp"
#include "copy_from_numpy_arrfunc.hpp"
#include "numpy_interop.hpp"
#include "utility_functions.hpp"

#include <dynd/functional.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/kernels/tuple_assignment_kernels.hpp>
#include <dynd/types/struct_type.hpp>

#include "kernels/copy_from_numpy_kernel.hpp"
#include "type_deduction.hpp"
#include "type_functions.hpp"
#include "types/pyobject_type.hpp"

using namespace std;

#ifdef DYND_NUMPY_INTEROP

namespace {

struct strided_of_numpy_arrmeta {
  dynd::fixed_dim_type_arrmeta sdt[NPY_MAXDIMS];
  pydynd::nd::copy_from_numpy_arrmeta am;
};

} // anonymous namespace

/*
void pydynd::nd::copy_from_numpy_callable::instantiate(dynd::nd::call_node *&node, char *DYND_UNUSED(data),
                                                       dynd::nd::kernel_builder *ckb, const dynd::ndt::type &dst_tp,
                                                       const char *dst_arrmeta, intptr_t DYND_UNUSED(nsrc),
                                                       const dynd::ndt::type *src_tp, const char *const *src_arrmeta,
                                                       dynd::kernel_request_t kernreq, intptr_t nkwd,
                                                       const dynd::nd::array *kwds,
                                                       const std::map<std::string, dynd::ndt::type> &tp_vars)
{
  if (src_tp[0].get_id() != dynd::void_id) {
    stringstream ss;
    ss << "Cannot instantiate dynd::nd::callable copy_from_numpy with "
          "signature (";
    ss << src_tp[0] << ") -> " << dst_tp;
    throw dynd::type_error(ss.str());
  }

  PyArray_Descr *dtype = *reinterpret_cast<PyArray_Descr *const *>(src_arrmeta[0]);
  uintptr_t src_alignment = reinterpret_cast<const uintptr_t *>(src_arrmeta[0])[1];

  if (!PyDataType_FLAGCHK(dtype, NPY_ITEM_HASOBJECT)) {
    // If there is no object type in the numpy type, get the dynd equivalent
    // type and use it to do the copying
    dynd::ndt::type src_view_tp = _type_from_numpy_dtype(dtype, src_alignment);
    dynd::nd::array error_mode = dynd::assign_error_fractional;
    dynd::nd::assign->instantiate(node, NULL, ckb, dst_tp, dst_arrmeta, 1, &src_view_tp, NULL, kernreq, 1,
                                  &error_mode, std::map<std::string, dynd::ndt::type>());

    return;
  }
  else if (PyDataType_ISOBJECT(dtype)) {
    dynd::nd::base_callable *af = dynd::nd::assign.get();
    dynd::ndt::type child_src_tp = dynd::ndt::make_type<pyobject_type>();
    af->instantiate(node, NULL, ckb, dst_tp, dst_arrmeta, 1, &child_src_tp, NULL, kernreq, nkwd, kwds, tp_vars);
    return;
  }
  else if (PyDataType_HASFIELDS(dtype)) {
    if (dst_tp.get_id() != dynd::struct_id && dst_tp.get_id() != dynd::tuple_id) {
      stringstream ss;
      ss << "Cannot assign from numpy type " << pyobject_repr((PyObject *)dtype) << " to dynd type " << dst_tp;
      throw invalid_argument(ss.str());
    }

    // Get the fields out of the numpy dtype
    vector<PyArray_Descr *> field_dtypes_orig;
    vector<std::string> field_names_orig;
    vector<size_t> field_offsets_orig;
    pydynd::extract_fields_from_numpy_struct(dtype, field_dtypes_orig, field_names_orig, field_offsets_orig);
    intptr_t field_count = field_dtypes_orig.size();
    if (field_count != dst_tp.extended<dynd::ndt::tuple_type>()->get_field_count()) {
      stringstream ss;
      ss << "Cannot assign from numpy type " << pyobject_repr((PyObject *)dtype) << " to dynd type " << dst_tp;
      throw invalid_argument(ss.str());
    }

    // Permute the numpy fields to match with the dynd fields
    vector<PyArray_Descr *> field_dtypes;
    vector<size_t> field_offsets;
    if (dst_tp.get_id() == dynd::struct_id) {
      field_dtypes.resize(field_count);
      field_offsets.resize(field_count);
      for (intptr_t i = 0; i < field_count; ++i) {
        intptr_t src_i = dst_tp.extended<dynd::ndt::struct_type>()->get_field_index(field_names_orig[i]);
        if (src_i >= 0) {
          field_dtypes[src_i] = field_dtypes_orig[i];
          field_offsets[src_i] = field_offsets_orig[i];
        }
        else {
          stringstream ss;
          ss << "Cannot assign from numpy type " << pyobject_repr((PyObject *)dtype) << " to dynd type " << dst_tp;
          throw invalid_argument(ss.str());
        }
      }
    }
    else {
      // In the tuple case, use position instead of name
      field_dtypes.swap(field_dtypes_orig);
      field_offsets.swap(field_offsets_orig);
    }

    vector<dynd::ndt::type> src_fields_tp(field_count, dynd::ndt::make_type<void>());
    vector<copy_from_numpy_arrmeta> src_arrmeta_values(field_count);
    vector<const char *> src_fields_arrmeta(field_count);
    for (intptr_t i = 0; i < field_count; ++i) {
      src_arrmeta_values[i].src_dtype = field_dtypes[i];
      src_arrmeta_values[i].src_alignment = src_alignment | field_offsets[i];
      src_fields_arrmeta[i] = reinterpret_cast<const char *>(&src_arrmeta_values[i]);
    }

    const uintptr_t *dst_arrmeta_offsets = dst_tp.extended<dynd::ndt::tuple_type>()->get_arrmeta_offsets_raw();
    dynd::shortvector<const char *> dst_fields_arrmeta(field_count);
    for (intptr_t i = 0; i != field_count; ++i) {
      dst_fields_arrmeta[i] = dst_arrmeta + dst_arrmeta_offsets[i];
    }

    // Todo: Remove this line
    dynd::nd::callable af = dynd::nd::make_callable<copy_from_numpy_callable>();

    const std::vector<dynd::ndt::type> &dst_fields_tp = dst_tp.extended<dynd::ndt::tuple_type>()->get_field_types();
    const uintptr_t *dst_data_offsets = dst_tp.extended<dynd::ndt::tuple_type>()->get_data_offsets(dst_arrmeta);

    intptr_t self_offset = ckb->size();
    ckb->emplace_back<dynd::nd::tuple_unary_op_ck>(kernreq);
    dynd::nd::tuple_unary_op_ck *self = ckb->get_at<dynd::nd::tuple_unary_op_ck>(self_offset);
    self->m_fields.resize(field_count);
    for (intptr_t i = 0; i < field_count; ++i) {
      self = ckb->get_at<dynd::nd::tuple_unary_op_ck>(self_offset);
      dynd::nd::tuple_unary_op_item &field = self->m_fields[i];
      field.child_kernel_offset = ckb->size() - self_offset;
      field.dst_data_offset = dst_data_offsets[i];
      field.src_data_offset = field_offsets[i];
      dynd::nd::array error_mode = dynd::ndt::traits<dynd::assign_error_mode>::na();
      af->instantiate(node, NULL, ckb, dst_fields_tp[i], dst_fields_arrmeta[i], 1, &src_fields_tp[i],
                      &src_fields_arrmeta[i], dynd::kernel_request_single, 1, &error_mode,
                      std::map<std::string, dynd::ndt::type>());
    }

    return;
  }
  else {
    stringstream ss;
    ss << "TODO: implement assign from numpy type " << pyobject_repr((PyObject *)dtype) << " to dynd type " << dst_tp;
    throw invalid_argument(ss.str());
  }
}
*/

dynd::nd::callable pydynd::nd::copy_from_numpy =
    dynd::nd::functional::elwise(dynd::nd::make_callable<pydynd::nd::copy_from_numpy_callable>());

void pydynd::nd::array_copy_from_numpy(const dynd::ndt::type &dst_tp, const char *dst_arrmeta, char *dst_data,
                                       PyArrayObject *src_arr, const dynd::eval::eval_context *ectx)
{
  intptr_t src_ndim = PyArray_NDIM(src_arr);

  strided_of_numpy_arrmeta src_am_holder;
  const char *src_am = reinterpret_cast<const char *>(&src_am_holder.sdt[NPY_MAXDIMS - src_ndim]);
  // Fill in metadata for a multi-dim strided array, corresponding
  // to the numpy array, with a void type at the end for the numpy
  // specific data.
  uintptr_t src_alignment = reinterpret_cast<uintptr_t>(PyArray_DATA(src_arr));
  for (intptr_t i = 0; i < src_ndim; ++i) {
    dynd::fixed_dim_type_arrmeta &am = src_am_holder.sdt[NPY_MAXDIMS - src_ndim + i];
    am.dim_size = PyArray_DIM(src_arr, (int)i);
    am.stride = am.dim_size != 1 ? PyArray_STRIDE(src_arr, (int)i) : 0;
    src_alignment |= static_cast<uintptr_t>(am.stride);
  }
  dynd::ndt::type src_tp = dynd::ndt::make_type(src_ndim, PyArray_SHAPE(src_arr), dynd::ndt::make_type<void>());
  src_am_holder.am.src_dtype = PyArray_DTYPE(src_arr);
  src_am_holder.am.src_alignment = src_alignment;

  char *src_data = reinterpret_cast<char *>(PyArray_DATA(src_arr));
  const char *kwd_names[1] = {"broadcast"};
  dynd::nd::array kwd_values[1] = {true};
  pydynd::nd::copy_from_numpy->call(dst_tp, dst_arrmeta, dst_data, 1, &src_tp, &src_am, &src_data, 1, kwd_values,
                                    std::map<std::string, dynd::ndt::type>());
}

#endif // DYND_NUMPY_INTEROP

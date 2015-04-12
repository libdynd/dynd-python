//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>
#include <datetime.h>

#include "copy_from_pyobject_arrfunc.hpp"
#include "copy_from_numpy_arrfunc.hpp"
#include "numpy_interop.hpp"
#include "utility_functions.hpp"
#include "type_functions.hpp"
#include "array_functions.hpp"
#include "array_from_py_typededuction.hpp"

#include "kernels/copy_from_pyobject_kernel.hpp"

#include <dynd/array.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/fixed_bytes_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/categorical_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/time_type.hpp>
#include <dynd/types/datetime_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/base_tuple_type.hpp>
#include <dynd/types/base_struct_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/func/copy.hpp>
#include <dynd/kernels/chain_kernel.hpp>
#include <dynd/func/chain.hpp>
#include <dynd/parser_util.hpp>

using namespace std;
using namespace pydynd;

static dynd::nd::arrfunc make_copy_from_pyobject_arrfunc(bool dim_broadcast);

static intptr_t instantiate_copy_from_pyobject(
    const arrfunc_type_data *self_af, const arrfunc_type *af_tp,
    char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
    const dynd::ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
    const dynd::ndt::type *src_tp, const char *const *src_arrmeta,
    kernel_request_t kernreq, const eval::eval_context *ectx,
    const dynd::nd::array &kwds,
    const std::map<dynd::nd::string, dynd::ndt::type> &tp_vars)
{
  if (src_tp[0].get_type_id() != void_type_id) {
    stringstream ss;
    ss << "Cannot instantiate arrfunc copy_from_pyobject with signature ";
    ss << af_tp << " with types (";
    ss << src_tp[0] << ") -> " << dst_tp;
    throw type_error(ss.str());
  }

  if (!kwds.is_null()) {
    throw invalid_argument("unexpected non-NULL kwds value to "
                           "copy_from_pyobject instantiation");
  }

  bool dim_broadcast = *self_af->get_data_as<bool>();

  switch (dst_tp.get_type_id()) {
  case bool_type_id:
    pydynd::nd::copy_from_pyobject_kernel<bool_type_id>::make(ckb, kernreq,
                                                              ckb_offset);
    return ckb_offset;
  case int8_type_id:
    pydynd::nd::int_copy_from_pyobject_kernel<int8_t>::make(ckb, kernreq,
                                                            ckb_offset);
    return ckb_offset;
  case int16_type_id:
    pydynd::nd::int_copy_from_pyobject_kernel<int16_t>::make(ckb, kernreq,
                                                             ckb_offset);
    return ckb_offset;
  case int32_type_id:
    pydynd::nd::int_copy_from_pyobject_kernel<int32_t>::make(ckb, kernreq,
                                                             ckb_offset);
    return ckb_offset;
  case int64_type_id:
    pydynd::nd::int_copy_from_pyobject_kernel<int64_t>::make(ckb, kernreq,
                                                             ckb_offset);
    return ckb_offset;
  case int128_type_id:
    pydynd::nd::int_copy_from_pyobject_kernel<dynd_int128>::make(ckb, kernreq,
                                                                 ckb_offset);
    return ckb_offset;
  case uint8_type_id:
    pydynd::nd::int_copy_from_pyobject_kernel<uint8_t>::make(ckb, kernreq,
                                                             ckb_offset);
    return ckb_offset;
  case uint16_type_id:
    pydynd::nd::int_copy_from_pyobject_kernel<uint16_t>::make(ckb, kernreq,
                                                              ckb_offset);
    return ckb_offset;
  case uint32_type_id:
    pydynd::nd::int_copy_from_pyobject_kernel<uint32_t>::make(ckb, kernreq,
                                                              ckb_offset);
    return ckb_offset;
  case uint64_type_id:
    pydynd::nd::int_copy_from_pyobject_kernel<uint64_t>::make(ckb, kernreq,
                                                              ckb_offset);
    return ckb_offset;
  case uint128_type_id:
    pydynd::nd::int_copy_from_pyobject_kernel<dynd_uint128>::make(ckb, kernreq,
                                                                  ckb_offset);
    return ckb_offset;
  case float16_type_id:
    pydynd::nd::float_copy_from_pyobject_kernel<dynd_float16>::make(
        ckb, kernreq, ckb_offset);
    return ckb_offset;
  case float32_type_id:
    pydynd::nd::float_copy_from_pyobject_kernel<float>::make(ckb, kernreq,
                                                             ckb_offset);
    return ckb_offset;
  case float64_type_id:
    pydynd::nd::float_copy_from_pyobject_kernel<double>::make(ckb, kernreq,
                                                              ckb_offset);
    return ckb_offset;
  case complex_float32_type_id:
    pydynd::nd::complex_float_copy_from_pyobject_kernel<float>::make(
        ckb, kernreq, ckb_offset);
    return ckb_offset;
  case complex_float64_type_id:
    pydynd::nd::complex_float_copy_from_pyobject_kernel<double>::make(
        ckb, kernreq, ckb_offset);
    return ckb_offset;
  case bytes_type_id:
  case fixed_bytes_type_id:
    pydynd::nd::bytes_copy_from_pyobject_kernel::make(ckb, kernreq, ckb_offset,
                                                      dst_tp, dst_arrmeta);
    return ckb_offset;
  case string_type_id:
  case fixed_string_type_id:
    pydynd::nd::string_copy_from_pyobject_kernel::make(ckb, kernreq, ckb_offset,
                                                       dst_tp, dst_arrmeta);
    return ckb_offset;
  case categorical_type_id: {
    // Assign via an intermediate category_type buffer
    const dynd::ndt::type &buf_tp =
        dst_tp.extended<categorical_type>()->get_category_type();
    dynd::nd::arrfunc copy_af =
        make_arrfunc_from_assignment(dst_tp, buf_tp, assign_error_default);
    dynd::nd::arrfunc af = dynd::nd::functional::chain(
        make_copy_from_pyobject_arrfunc(dim_broadcast), copy_af, buf_tp);
    return af.get()->instantiate(af.get(), af.get_type(), NULL, ckb, ckb_offset,
                                 dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta,
                                 kernreq, ectx, kwds, tp_vars);
  }

  case date_type_id:
    pydynd::nd::copy_from_pyobject_kernel<date_type_id>::make(
        ckb, kernreq, ckb_offset, dst_tp, dst_arrmeta);
    return ckb_offset;
  case time_type_id:
    pydynd::nd::copy_from_pyobject_kernel<time_type_id>::make(
        ckb, kernreq, ckb_offset, dst_tp, dst_arrmeta);
    return ckb_offset;
  case datetime_type_id:
    pydynd::nd::copy_from_pyobject_kernel<datetime_type_id>::make(
        ckb, kernreq, ckb_offset, dst_tp, dst_arrmeta);
    return ckb_offset;
  case type_type_id:
    pydynd::nd::copy_from_pyobject_kernel<type_type_id>::make(ckb, kernreq,
                                                              ckb_offset);
    return ckb_offset;
  case option_type_id: {
    intptr_t root_ckb_offset = ckb_offset;
    pydynd::nd::copy_from_pyobject_kernel<option_type_id> *self =
        pydynd::nd::copy_from_pyobject_kernel<option_type_id>::make(
            ckb, kernreq, ckb_offset);
    self->m_dst_tp = dst_tp;
    self->m_dst_arrmeta = dst_arrmeta;
    const arrfunc_type_data *assign_na_af =
        dst_tp.extended<option_type>()->get_assign_na_arrfunc();
    const arrfunc_type *assign_na_af_tp =
        dst_tp.extended<option_type>()->get_assign_na_arrfunc_type();
    ckb_offset = assign_na_af->instantiate(
        assign_na_af, assign_na_af_tp, NULL, ckb, ckb_offset, dst_tp,
        dst_arrmeta, nsrc, NULL, NULL, kernel_request_single, ectx,
        dynd::nd::array(), tp_vars);
    reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
        ->reserve(ckb_offset);
    self = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
               ->get_at<pydynd::nd::copy_from_pyobject_kernel<option_type_id>>(
                   root_ckb_offset);
    self->m_copy_value_offset = ckb_offset - root_ckb_offset;
    ckb_offset = self_af->instantiate(
        self_af, af_tp, NULL, ckb, ckb_offset,
        dst_tp.extended<option_type>()->get_value_type(), dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernel_request_single, ectx, dynd::nd::array(),
        tp_vars);
    return ckb_offset;
  }
  case fixed_dim_type_id: {
    intptr_t dim_size, stride;
    dynd::ndt::type el_tp;
    const char *el_arrmeta;
    if (dst_tp.get_as_strided(dst_arrmeta, &dim_size, &stride, &el_tp,
                              &el_arrmeta)) {
      intptr_t root_ckb_offset = ckb_offset;
      pydynd::nd::copy_from_pyobject_kernel<fixed_dim_type_id> *self = pydynd::nd::copy_from_pyobject_kernel<fixed_dim_type_id>::make(ckb, kernreq, ckb_offset);
      self->m_dim_size = dim_size;
      self->m_stride = stride;
      self->m_dst_tp = dst_tp;
      self->m_dst_arrmeta = dst_arrmeta;
      self->m_dim_broadcast = dim_broadcast;
      // from pyobject ckernel
      ckb_offset = self_af->instantiate(self_af, af_tp, NULL, ckb, ckb_offset,
                                        el_tp, el_arrmeta, nsrc, src_tp,
                                        src_arrmeta, kernel_request_strided,
                                        ectx, dynd::nd::array(), tp_vars);
      self = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
                 ->get_at<pydynd::nd::copy_from_pyobject_kernel<fixed_dim_type_id>>(root_ckb_offset);
      self->m_copy_dst_offset = ckb_offset - root_ckb_offset;
      // dst to dst ckernel, for broadcasting case
      return make_assignment_kernel(
          NULL, NULL, ckb, ckb_offset, el_tp, el_arrmeta, el_tp, el_arrmeta,
          kernel_request_strided, ectx, dynd::nd::array());
    }
    break;
  }
  case var_dim_type_id: {
    intptr_t root_ckb_offset = ckb_offset;
    pydynd::nd::copy_from_pyobject_kernel<var_dim_type_id> *self = pydynd::nd::copy_from_pyobject_kernel<var_dim_type_id>::make(ckb, kernreq, ckb_offset);
    self->m_offset =
        reinterpret_cast<const var_dim_type_arrmeta *>(dst_arrmeta)->offset;
    self->m_stride =
        reinterpret_cast<const var_dim_type_arrmeta *>(dst_arrmeta)->stride;
    self->m_dst_tp = dst_tp;
    self->m_dst_arrmeta = dst_arrmeta;
    self->m_dim_broadcast = dim_broadcast;
    dynd::ndt::type el_tp = dst_tp.extended<var_dim_type>()->get_element_type();
    const char *el_arrmeta = dst_arrmeta + sizeof(var_dim_type_arrmeta);
    ckb_offset = self_af->instantiate(
        self_af, af_tp, NULL, ckb, ckb_offset, el_tp, el_arrmeta, nsrc, src_tp,
        src_arrmeta, kernel_request_strided, ectx, dynd::nd::array(), tp_vars);
    self = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
               ->get_at<pydynd::nd::copy_from_pyobject_kernel<var_dim_type_id>>(root_ckb_offset);
    self->m_copy_dst_offset = ckb_offset - root_ckb_offset;
    // dst to dst ckernel, for broadcasting case
    return make_assignment_kernel(
        NULL, NULL, ckb, ckb_offset, el_tp, el_arrmeta, el_tp, el_arrmeta,
        kernel_request_strided, ectx, dynd::nd::array());
  }
  case tuple_type_id: {
    intptr_t root_ckb_offset = ckb_offset;
    pydynd::nd::copy_from_pyobject_kernel<tuple_type_id> *self = pydynd::nd::copy_from_pyobject_kernel<tuple_type_id>::make(ckb, kernreq, ckb_offset);
    self->m_dst_tp = dst_tp;
    self->m_dst_arrmeta = dst_arrmeta;
    intptr_t field_count =
        dst_tp.extended<base_tuple_type>()->get_field_count();
    const dynd::ndt::type *field_types =
        dst_tp.extended<base_tuple_type>()->get_field_types_raw();
    const uintptr_t *arrmeta_offsets =
        dst_tp.extended<base_tuple_type>()->get_arrmeta_offsets_raw();
    self->m_dim_broadcast = dim_broadcast;
    self->m_copy_el_offsets.resize(field_count);
    for (intptr_t i = 0; i < field_count; ++i) {
      reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
          ->reserve(ckb_offset);
      self = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
                 ->get_at<pydynd::nd::copy_from_pyobject_kernel<tuple_type_id>>(root_ckb_offset);
      self->m_copy_el_offsets[i] = ckb_offset - root_ckb_offset;
      const char *field_arrmeta = dst_arrmeta + arrmeta_offsets[i];
      ckb_offset = self_af->instantiate(
          self_af, af_tp, NULL, ckb, ckb_offset, field_types[i], field_arrmeta,
          nsrc, src_tp, src_arrmeta, kernel_request_single, ectx,
          dynd::nd::array(), tp_vars);
    }
    return ckb_offset;
  }
  case struct_type_id: {
    intptr_t root_ckb_offset = ckb_offset;
    pydynd::nd::copy_from_pyobject_kernel<struct_type_id> *self = pydynd::nd::copy_from_pyobject_kernel<struct_type_id>::make(ckb, kernreq, ckb_offset);
    self->m_dst_tp = dst_tp;
    self->m_dst_arrmeta = dst_arrmeta;
    intptr_t field_count =
        dst_tp.extended<base_struct_type>()->get_field_count();
    const dynd::ndt::type *field_types =
        dst_tp.extended<base_struct_type>()->get_field_types_raw();
    const uintptr_t *arrmeta_offsets =
        dst_tp.extended<base_struct_type>()->get_arrmeta_offsets_raw();
    self->m_dim_broadcast = dim_broadcast;
    self->m_copy_el_offsets.resize(field_count);
    for (intptr_t i = 0; i < field_count; ++i) {
      reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
          ->reserve(ckb_offset);
      self = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
                 ->get_at<pydynd::nd::copy_from_pyobject_kernel<struct_type_id>>(root_ckb_offset);
      self->m_copy_el_offsets[i] = ckb_offset - root_ckb_offset;
      const char *field_arrmeta = dst_arrmeta + arrmeta_offsets[i];
      ckb_offset = self_af->instantiate(
          self_af, af_tp, NULL, ckb, ckb_offset, field_types[i], field_arrmeta,
          nsrc, src_tp, src_arrmeta, kernel_request_single, ectx,
          dynd::nd::array(), tp_vars);
    }
    return ckb_offset;
  }
  default:
    break;
  }

  if (dst_tp.get_kind() == expr_kind) {
    dynd::nd::arrfunc af = dynd::nd::functional::chain(
        make_copy_from_pyobject_arrfunc(dim_broadcast), dynd::nd::copy,
        dst_tp.value_type());
    return af.get()->instantiate(af.get(), af.get_type(), NULL, ckb, ckb_offset,
                                 dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta,
                                 kernreq, ectx, kwds, tp_vars);
  }

  stringstream ss;
  ss << "Unable to copy a Python object to dynd value with type " << dst_tp;
  throw invalid_argument(ss.str());
}

static dynd::nd::arrfunc make_copy_from_pyobject_arrfunc(bool dim_broadcast)
{
  dynd::nd::array out_af = dynd::nd::empty("(void) -> A... * T");
  arrfunc_type_data *af =
      reinterpret_cast<arrfunc_type_data *>(out_af.get_readwrite_originptr());
  af->instantiate = &instantiate_copy_from_pyobject;
  *af->get_data_as<bool>() = dim_broadcast;
  out_af.flag_as_immutable();
  return out_af;
}

dynd::nd::arrfunc pydynd::copy_from_pyobject::make()
{
  PyDateTime_IMPORT;
  return make_copy_from_pyobject_arrfunc(true);
}

dynd::nd::arrfunc pydynd::copy_from_pyobject_no_dim_broadcast::make()
{
  PyDateTime_IMPORT;
  return make_copy_from_pyobject_arrfunc(false);
}

struct pydynd::copy_from_pyobject pydynd::copy_from_pyobject;
struct pydynd::copy_from_pyobject_no_dim_broadcast
    pydynd::copy_from_pyobject_no_dim_broadcast;
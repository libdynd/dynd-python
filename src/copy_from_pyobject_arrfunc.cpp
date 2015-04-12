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

intptr_t pydynd::nd::copy_from_pyobject_virtual_kernel::instantiate(
    const arrfunc_type_data *self_af, const arrfunc_type *af_tp, char *data,
    void *ckb, intptr_t ckb_offset, const dynd::ndt::type &dst_tp,
    const char *dst_arrmeta, intptr_t nsrc, const dynd::ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx, const dynd::nd::array &kwds,
    const std::map<dynd::nd::string, dynd::ndt::type> &tp_vars)
{
  if (src_tp[0].get_type_id() != void_type_id) {
    stringstream ss;
    ss << "Cannot instantiate arrfunc copy_from_pyobject with signature ";
    ss << af_tp << " with types (";
    ss << src_tp[0] << ") -> " << dst_tp;
    throw type_error(ss.str());
  }

  bool dim_broadcast = *self_af->get_data_as<bool>();

  switch (dst_tp.get_type_id()) {
  case bool_type_id:
    return copy_from_pyobject_kernel<bool_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case int8_type_id:
    return copy_from_pyobject_kernel<int8_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case int16_type_id:
    return copy_from_pyobject_kernel<int16_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case int32_type_id:
    return copy_from_pyobject_kernel<int32_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case int64_type_id:
    return copy_from_pyobject_kernel<int64_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case int128_type_id:
    return copy_from_pyobject_kernel<int128_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case uint8_type_id:
    return copy_from_pyobject_kernel<uint8_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case uint16_type_id:
    return copy_from_pyobject_kernel<uint16_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case uint32_type_id:
    return copy_from_pyobject_kernel<uint32_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case uint64_type_id:
    return copy_from_pyobject_kernel<uint64_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case uint128_type_id:
    return copy_from_pyobject_kernel<uint128_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case float16_type_id:
    return copy_from_pyobject_kernel<float16_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case float32_type_id:
    return copy_from_pyobject_kernel<float32_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case float64_type_id:
    return copy_from_pyobject_kernel<float64_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case complex_float32_type_id:
    return copy_from_pyobject_kernel<complex_float32_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case complex_float64_type_id:
    return copy_from_pyobject_kernel<complex_float64_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case bytes_type_id:
    return copy_from_pyobject_kernel<bytes_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case fixed_bytes_type_id:
    return copy_from_pyobject_kernel<fixed_bytes_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case string_type_id:
    return copy_from_pyobject_kernel<string_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case fixed_string_type_id:
    return copy_from_pyobject_kernel<fixed_string_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case categorical_type_id:
    return copy_from_pyobject_kernel<categorical_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case date_type_id:
    return copy_from_pyobject_kernel<date_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case time_type_id:
    return copy_from_pyobject_kernel<time_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case datetime_type_id:
    return copy_from_pyobject_kernel<datetime_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case type_type_id:
    return copy_from_pyobject_kernel<type_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case option_type_id:
    return copy_from_pyobject_kernel<option_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case fixed_dim_type_id:
    return copy_from_pyobject_kernel<fixed_dim_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case var_dim_type_id:
    return copy_from_pyobject_kernel<var_dim_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case tuple_type_id:
    return copy_from_pyobject_kernel<tuple_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case struct_type_id:
    return copy_from_pyobject_kernel<struct_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  default:
    break;
  }

  return default_copy_from_pyobject_kernel::instantiate(
      self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, src_tp,
      src_arrmeta, kernreq, ectx, kwds, tp_vars);

  stringstream ss;
  ss << "Unable to copy a Python object to dynd value with type " << dst_tp;
  throw invalid_argument(ss.str());
}

static dynd::nd::arrfunc make_copy_from_pyobject_arrfunc(bool dim_broadcast)
{
  return dynd::nd::as_arrfunc<pydynd::nd::copy_from_pyobject_virtual_kernel>(
      dynd::ndt::type("(void) -> Any"), dim_broadcast, 0);
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
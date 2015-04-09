//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>
#include <datetime.h>

#include "copy_to_pyobject_arrfunc.hpp"
#include "utility_functions.hpp"
#include "type_functions.hpp"

#include <dynd/array.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/fixedbytes_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/categorical_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/time_type.hpp>
#include <dynd/types/datetime_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/base_tuple_type.hpp>
#include <dynd/types/base_struct_type.hpp>
#include <dynd/types/pointer_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/func/copy.hpp>
#include <dynd/kernels/chain.hpp>
#include <dynd/func/chain.hpp>

#include "kernels/copy_kernel.hpp"

using namespace std;
using namespace pydynd;

namespace pydynd {
namespace nd {

  static dynd::nd::arrfunc
  make_copy_to_pyobject_arrfunc(bool struct_as_pytuple);
}
}

intptr_t pydynd::nd::copy_to_pyobject_virtual_kernel::instantiate(
    const arrfunc_type_data *self_af, const arrfunc_type *af_tp, char *data,
    void *ckb, intptr_t ckb_offset, const dynd::ndt::type &dst_tp,
    const char *dst_arrmeta, intptr_t nsrc, const dynd::ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx, const dynd::nd::array &kwds,
    const std::map<dynd::nd::string, dynd::ndt::type> &tp_vars)
{
  if (dst_tp.get_type_id() != void_type_id) {
    stringstream ss;
    ss << "Cannot instantiate arrfunc with signature ";
    ss << af_tp << " with types (";
    ss << src_tp[0] << ") -> " << dst_tp;
    throw type_error(ss.str());
  }

  if (!kwds.is_null()) {
    throw invalid_argument("unexpected non-NULL kwds value to "
                           "copy_to_pyobject instantiation");
  }

  bool struct_as_pytuple = *self_af->get_data_as<bool>();

  switch (src_tp[0].get_type_id()) {
  case bool_type_id:
    return copy_to_pyobject_kernel<bool_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case int8_type_id:
    return copy_to_pyobject_kernel<int8_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case int16_type_id:
    return copy_to_pyobject_kernel<int16_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case int32_type_id:
    return copy_to_pyobject_kernel<int32_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case int64_type_id:
    return copy_to_pyobject_kernel<int64_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case int128_type_id:
    return copy_to_pyobject_kernel<int128_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case uint8_type_id:
    return copy_to_pyobject_kernel<uint8_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case uint16_type_id:
    return copy_to_pyobject_kernel<uint16_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case uint32_type_id:
    return copy_to_pyobject_kernel<uint32_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case uint64_type_id:
    return copy_to_pyobject_kernel<uint64_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case uint128_type_id:
    return copy_to_pyobject_kernel<uint128_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case float16_type_id:
    return copy_to_pyobject_kernel<float16_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case float32_type_id:
    return copy_to_pyobject_kernel<float32_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case float64_type_id:
    return copy_to_pyobject_kernel<float64_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case complex_float32_type_id:
    return copy_to_pyobject_kernel<complex_float32_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case complex_float64_type_id:
    return copy_to_pyobject_kernel<complex_float64_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case bytes_type_id:
    return copy_to_pyobject_kernel<bytes_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case fixedbytes_type_id:
    return copy_to_pyobject_kernel<fixedbytes_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case char_type_id:
    return copy_to_pyobject_kernel<char_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case string_type_id:
    return copy_to_pyobject_kernel<string_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case fixedstring_type_id:
    return copy_to_pyobject_kernel<fixedstring_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case categorical_type_id: {
    // Assign via an intermediate category_type buffer
    const dynd::ndt::type &buf_tp =
        src_tp[0].extended<categorical_type>()->get_category_type();
    dynd::nd::arrfunc copy_af =
        make_arrfunc_from_assignment(buf_tp, src_tp[0], assign_error_default);
    dynd::nd::arrfunc af = dynd::nd::functional::chain(
        copy_af, pydynd::nd::make_copy_to_pyobject_arrfunc(struct_as_pytuple),
        buf_tp);
    return af.get()->instantiate(af.get(), af.get_type(), NULL, ckb, ckb_offset,
                                 dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta,
                                 kernreq, ectx, kwds, tp_vars);
  }
  case date_type_id:
    return copy_to_pyobject_kernel<date_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case time_type_id:
    return copy_to_pyobject_kernel<time_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case datetime_type_id:
    return copy_to_pyobject_kernel<datetime_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case type_type_id:
    return copy_to_pyobject_kernel<type_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case option_type_id:
    return copy_to_pyobject_kernel<option_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case fixed_dim_type_id:
  case cfixed_dim_type_id:
    return copy_to_pyobject_kernel<fixed_dim_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case var_dim_type_id:
    return copy_to_pyobject_kernel<var_dim_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case cstruct_type_id:
  case struct_type_id:
    if (!struct_as_pytuple) {
      return copy_to_pyobject_kernel<struct_type_id>::instantiate(
          self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
          src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
    }
  // Otherwise fall through to the tuple case
  case ctuple_type_id:
  case tuple_type_id:
    return copy_to_pyobject_kernel<tuple_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case pointer_type_id:
    return copy_to_pyobject_kernel<pointer_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case groupby_type_id:
  case expr_type_id:
  case convert_type_id:
  case property_type_id:
  case byteswap_type_id:
  case 44:
  case view_type_id:
  case unary_expr_type_id: {
    dynd::nd::arrfunc af = dynd::nd::functional::chain(
        dynd::nd::copy,
        pydynd::nd::make_copy_to_pyobject_arrfunc(struct_as_pytuple),
        src_tp[0].value_type());
    return af.get()->instantiate(af.get(), af.get_type(), NULL, ckb, ckb_offset,
                                 dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta,
                                 kernreq, ectx, kwds, tp_vars);
  }
  default:
    break;
  }

  stringstream ss;
  ss << "Unable to copy dynd value with type " << src_tp[0]
     << " to a Python object";
  throw invalid_argument(ss.str());
}

namespace pydynd {
namespace nd {

  static dynd::nd::arrfunc make_copy_to_pyobject_arrfunc(bool struct_as_pytuple)
  {
    return dynd::nd::as_arrfunc<pydynd::nd::copy_to_pyobject_virtual_kernel>(
        dynd::ndt::type("(Any) -> void"), struct_as_pytuple, 0);
  }
}
}

dynd::nd::arrfunc pydynd::nd::copy_to_pyobject_dict::make()
{
  PyDateTime_IMPORT;

  return pydynd::nd::make_copy_to_pyobject_arrfunc(false);
}

struct pydynd::nd::copy_to_pyobject_dict pydynd::nd::copy_to_pyobject_dict;

dynd::nd::arrfunc pydynd::nd::copy_to_pyobject_tuple::make()
{
  PyDateTime_IMPORT;
  return pydynd::nd::make_copy_to_pyobject_arrfunc(true);
}

struct pydynd::nd::copy_to_pyobject_tuple pydynd::nd::copy_to_pyobject_tuple;
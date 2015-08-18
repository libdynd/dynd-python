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
#include <dynd/types/pointer_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/func/copy.hpp>
#include <dynd/func/multidispatch.hpp>
#include <dynd/func/compose.hpp>

#include "kernels/copy_to_pyobject_kernel.hpp"

using namespace std;

typedef dynd::integer_sequence<
    dynd::type_id_t, dynd::bool_type_id, dynd::int8_type_id,
    dynd::int16_type_id, dynd::int32_type_id, dynd::int64_type_id,
    dynd::int128_type_id, dynd::uint8_type_id, dynd::uint16_type_id,
    dynd::uint32_type_id, dynd::uint64_type_id, dynd::uint128_type_id,
    dynd::float16_type_id, dynd::float32_type_id, dynd::float64_type_id,
    dynd::complex_float32_type_id, dynd::complex_float64_type_id,
    dynd::bytes_type_id, dynd::fixed_bytes_type_id, dynd::char_type_id,
    dynd::string_type_id, dynd::fixed_string_type_id, dynd::date_type_id,
    dynd::time_type_id, dynd::datetime_type_id, dynd::type_type_id,
    dynd::option_type_id, dynd::fixed_dim_type_id, dynd::var_dim_type_id,
    dynd::struct_type_id, dynd::tuple_type_id, dynd::pointer_type_id,
    dynd::categorical_type_id> type_ids;

dynd::nd::callable pydynd::nd::copy_to_pyobject::make()
{
  PyDateTime_IMPORT;

  std::map<dynd::type_id_t, dynd::nd::callable> callables =
      dynd::nd::callable::make_all<copy_to_pyobject_kernel, type_ids>(0);
  for (std::pair<dynd::type_id_t, dynd::nd::callable> pair : callables) {
    children[pair.first] = pair.second;
  }

  default_child = dynd::nd::callable::make<default_copy_to_pyobject_kernel>(0);

  return dynd::nd::functional::multidispatch(
      dynd::ndt::type("(Any) -> void"),
      [](const dynd::ndt::type &DYND_UNUSED(dst_tp), intptr_t DYND_UNUSED(nsrc),
         const dynd::ndt::type *src_tp) -> dynd::nd::callable & {
        dynd::nd::callable &child = overload(src_tp[0]);
        if (child.is_null()) {
          return default_child;
        }
        return child;
      },
      0);
}

struct pydynd::nd::copy_to_pyobject pydynd::nd::copy_to_pyobject;

dynd::nd::callable pydynd::nd::copy_to_pyobject::children[DYND_TYPE_ID_MAX + 1];
dynd::nd::callable pydynd::nd::copy_to_pyobject::default_child;

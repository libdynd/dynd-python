//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>
#include <datetime.h>

#include "copy_from_pyobject_arrfunc.hpp"
#include "copy_from_numpy_arrfunc.hpp"

#include "kernels/assign_from_pyobject_kernel.hpp"

using namespace std;
using namespace dynd;

PYDYND_API void assign_init()
{
  typedef type_id_sequence<
      bool_type_id, int8_type_id, int16_type_id, int32_type_id, int64_type_id,
      int128_type_id, uint8_type_id, uint16_type_id, uint32_type_id,
      uint64_type_id, uint128_type_id, float16_type_id, float32_type_id,
      float64_type_id, complex_float32_type_id, complex_float64_type_id,
      bytes_type_id, fixed_bytes_type_id, string_type_id, fixed_string_type_id,
      date_type_id, time_type_id, datetime_type_id, option_type_id,
      type_type_id, tuple_type_id, struct_type_id, fixed_dim_type_id,
      var_dim_type_id, categorical_type_id> type_ids;

  PyDateTime_IMPORT;

  nd::callable &assign = nd::assign::get();
  for (const auto &pair :
       nd::callable::make_all<assign_from_pyobject_kernel, type_ids>()) {
    assign.set_overload(pair.first, {ndt::make_type<pyobject_type>()},
                        pair.second);
  }
}

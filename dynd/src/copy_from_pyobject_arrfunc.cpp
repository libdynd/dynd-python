//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>
#include <datetime.h>

#include <dynd/functional.hpp>

#include "copy_from_pyobject_arrfunc.hpp"
#include "copy_from_numpy_arrfunc.hpp"
#include "numpy_interop.hpp"
#include "utility_functions.hpp"
#include "type_functions.hpp"
#include "array_functions.hpp"
#include "array_from_py_typededuction.hpp"

#include "kernels/copy_from_pyobject_kernel.hpp"

using namespace std;

dynd::nd::callable pydynd::nd::copy_from_pyobject::make()
{
  typedef dynd::type_id_sequence<
      dynd::bool_type_id, dynd::int8_type_id, dynd::int16_type_id,
      dynd::int32_type_id, dynd::int64_type_id, dynd::int128_type_id,
      dynd::uint8_type_id, dynd::uint16_type_id, dynd::uint32_type_id,
      dynd::uint64_type_id, dynd::uint128_type_id, dynd::float16_type_id,
      dynd::float32_type_id, dynd::float64_type_id,
      dynd::complex_float32_type_id, dynd::complex_float64_type_id,
      dynd::bytes_type_id, dynd::fixed_bytes_type_id, dynd::string_type_id,
      dynd::fixed_string_type_id, dynd::date_type_id, dynd::time_type_id,
      dynd::datetime_type_id, dynd::type_type_id, dynd::option_type_id,
      dynd::categorical_type_id, dynd::fixed_dim_type_id, dynd::var_dim_type_id,
      dynd::tuple_type_id, dynd::struct_type_id> I;

  PyDateTime_IMPORT;

  // ...
  auto children = dynd::nd::callable::make_all<copy_from_pyobject_kernel, I>();

  // ...
  dynd::nd::callable default_child =
      dynd::nd::callable::make<default_copy_from_pyobject_kernel>();

  return dynd::nd::functional::dispatch(
      dynd::ndt::type("(void, broadcast: bool) -> Any"),
      [children, default_child](
          const dynd::ndt::type &dst_tp, intptr_t DYND_UNUSED(nsrc),
          const dynd::ndt::type *src_tp) mutable -> dynd::nd::callable & {
        dynd::nd::callable &child = children[dst_tp.get_type_id()];
        if (child.is_null()) {
          return default_child;
        }
        return child;
      });
}

struct pydynd::nd::copy_from_pyobject pydynd::nd::copy_from_pyobject;

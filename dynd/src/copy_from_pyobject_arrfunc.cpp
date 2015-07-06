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

#include <dynd/func/multidispatch.hpp>

using namespace std;

dynd::nd::arrfunc pydynd::nd::copy_from_pyobject::make()
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
  for (const pair<const dynd::type_id_t, dynd::nd::arrfunc> &pair :
       dynd::nd::arrfunc::make_all<copy_from_pyobject_kernel, I>()) {
    children[pair.first] = pair.second;
  }

  // ...
  default_child = dynd::nd::arrfunc::make<default_copy_from_pyobject_kernel>(0);

  return dynd::nd::functional::multidispatch_by_type_id(
      dynd::ndt::type("(void, broadcast: bool) -> Any"), DYND_TYPE_ID_MAX + 1,
      children, default_child, -1);
}

struct pydynd::nd::copy_from_pyobject pydynd::nd::copy_from_pyobject;

dynd::nd::arrfunc
    pydynd::nd::copy_from_pyobject::children[DYND_TYPE_ID_MAX + 1];
dynd::nd::arrfunc pydynd::nd::copy_from_pyobject::default_child;
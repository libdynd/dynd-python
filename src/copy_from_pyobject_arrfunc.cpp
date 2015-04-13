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
using namespace pydynd;

dynd::nd::arrfunc pydynd::nd::copy_from_pyobject::make()
{
  typedef type_id_sequence<
      bool_type_id, int8_type_id, int16_type_id, int32_type_id, int64_type_id,
      int128_type_id, uint8_type_id, uint16_type_id, uint32_type_id,
      uint64_type_id, uint128_type_id, float16_type_id, float32_type_id,
      float64_type_id, complex_float32_type_id, complex_float64_type_id,
      bytes_type_id, fixed_bytes_type_id, string_type_id, fixed_string_type_id,
      date_type_id, time_type_id, datetime_type_id, type_type_id,
      option_type_id, categorical_type_id, fixed_dim_type_id, var_dim_type_id,
      tuple_type_id, struct_type_id> I;

  PyDateTime_IMPORT;

  arrfunc::make_all<copy_from_pyobject_kernel, I>(children);
  arrfunc::make<default_copy_from_pyobject_kernel>(default_child, 0);

  return functional::multidispatch_by_type_id(
      ndt::type("(void, broadcast: bool) -> Any"), DYND_TYPE_ID_MAX + 1,
      children, default_child, false, -1);
}

struct pydynd::nd::copy_from_pyobject pydynd::nd::copy_from_pyobject;

dynd::nd::arrfunc
    pydynd::nd::copy_from_pyobject::children[DYND_TYPE_ID_MAX + 1];
dynd::nd::arrfunc pydynd::nd::copy_from_pyobject::default_child;
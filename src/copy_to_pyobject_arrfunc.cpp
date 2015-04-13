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
#include <dynd/kernels/chain_kernel.hpp>
#include <dynd/func/chain.hpp>

#include "kernels/copy_to_pyobject_kernel.hpp"

using namespace std;
using namespace pydynd;

typedef integer_sequence<
    type_id_t, bool_type_id, int8_type_id, int16_type_id, int32_type_id,
    int64_type_id, int128_type_id, uint8_type_id, uint16_type_id,
    uint32_type_id, uint64_type_id, uint128_type_id, float16_type_id,
    float32_type_id, float64_type_id, complex_float32_type_id,
    complex_float64_type_id, bytes_type_id, fixed_bytes_type_id, char_type_id,
    string_type_id, fixed_string_type_id, date_type_id, time_type_id,
    datetime_type_id, type_type_id, option_type_id, fixed_dim_type_id,
    var_dim_type_id, struct_type_id, tuple_type_id, pointer_type_id,
    categorical_type_id> type_ids;

dynd::nd::arrfunc pydynd::nd::copy_to_pyobject::make()
{
  PyDateTime_IMPORT;

  arrfunc::make_all<copy_to_pyobject_kernel, type_ids>(children);
  arrfunc::make<default_copy_to_pyobject_kernel>(default_child, 0);

  return functional::multidispatch_by_type_id(ndt::type("(Any) -> void"),
                                              DYND_TYPE_ID_MAX + 1, children,
                                              default_child, false);
}

struct pydynd::nd::copy_to_pyobject pydynd::nd::copy_to_pyobject;

dynd::nd::arrfunc pydynd::nd::copy_to_pyobject::children[DYND_TYPE_ID_MAX + 1];
dynd::nd::arrfunc pydynd::nd::copy_to_pyobject::default_child;

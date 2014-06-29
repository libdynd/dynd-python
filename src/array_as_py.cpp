//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>

#include "array_as_py.hpp"
#include "array_functions.hpp"
#include "type_functions.hpp"
#include "utility_functions.hpp"
#include "copy_to_pyobject_ckernel.hpp"

#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/base_struct_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/time_type.hpp>
#include <dynd/types/datetime_type.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/type_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dynd;
using namespace pydynd;

PyObject* pydynd::array_as_py(const dynd::nd::array& a, bool struct_as_pytuple)
{
  // Evaluate the nd::array
  assignment_ckernel_builder ckb;
  make_copy_to_pyobject_kernel(&ckb, 0, a.get_type(), a.get_arrmeta(),
                               struct_as_pytuple, kernel_request_single,
                               &eval::default_eval_context);
  pyobject_ownref result;
  ckb(reinterpret_cast<char *>(result.obj_addr()), a.get_readonly_originptr());
  return result.release();
}


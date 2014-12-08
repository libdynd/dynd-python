//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>

#include "array_as_py.hpp"
#include "array_functions.hpp"
#include "type_functions.hpp"
#include "utility_functions.hpp"
#include "copy_to_pyobject_arrfunc.hpp"

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

PyObject *pydynd::array_as_py(const dynd::nd::array &a, bool struct_as_pytuple)
{
  // Evaluate the nd::array
  unary_ckernel_builder ckb;
  const arrfunc_type_data *af;
  const arrfunc_type *af_tp;
  if (struct_as_pytuple) {
    af = copy_to_pyobject_tuple.get();
    af_tp = copy_to_pyobject_tuple.get_type();
  } else {
    af = copy_to_pyobject_dict.get();
    af_tp = copy_to_pyobject_dict.get_type();
  }
  ndt::type tp = a.get_type();
  const char *arrmeta = a.get_arrmeta();
  af->instantiate(af, af_tp, &ckb, 0, ndt::make_type<void>(), NULL, &tp,
                  &arrmeta, kernel_request_single, &eval::default_eval_context,
                  nd::array());
  pyobject_ownref result;
  ckb(reinterpret_cast<char *>(result.obj_addr()),
      const_cast<char *>(a.get_readonly_originptr()));
  if (PyErr_Occurred()) {
    throw exception();
  }
  return result.release();
}


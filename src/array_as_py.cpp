//
// Copyright (C) 2011-15 DyND Developers
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

using namespace std;
using namespace dynd;
using namespace pydynd;

PyObject *pydynd::array_as_py(const dynd::nd::array &a, bool struct_as_pytuple)
{
  // Evaluate the nd::array
  ckernel_builder<kernel_request_host> ckb;
  const arrfunc_type_data *af;
  const arrfunc_type *af_tp;
  af = nd::copy_to_pyobject.get();
  af_tp = nd::copy_to_pyobject.get_type();
  ndt::type tp = a.get_type();
  const char *arrmeta = a.get_arrmeta();
  af->instantiate(af, af_tp, NULL, &ckb, 0, ndt::make_type<void>(), NULL, 1,
                  &tp, &arrmeta, kernel_request_single,
                  &eval::default_eval_context, dynd::nd::array(),
                  std::map<dynd::nd::string, ndt::type>());
  pyobject_ownref result;
  expr_single_t fn = ckb.get()->get_function<expr_single_t>();
  char *src = const_cast<char *>(a.get_readonly_originptr());
  fn(reinterpret_cast<char *>(result.obj_addr()), &src, ckb.get());
  if (PyErr_Occurred()) {
    throw exception();
  }
  return result.release();
}

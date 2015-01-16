//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>

#include <dynd/kernels/assignment_kernels.hpp>

#include "array_from_py.hpp"
#include "array_assign_from_py.hpp"
#include "array_functions.hpp"
#include "type_functions.hpp"
#include "utility_functions.hpp"
#include "numpy_interop.hpp"
#include "copy_from_pyobject_arrfunc.hpp"

using namespace std;
using namespace dynd;
using namespace pydynd;

void pydynd::array_broadcast_assign_from_py(const dynd::ndt::type &dt,
                                            const char *arrmeta, char *data,
                                            PyObject *value,
                                            const eval::eval_context *ectx)
{
  unary_ckernel_builder ckb;
  const arrfunc_type_data *af = copy_from_pyobject.get();
  ndt::type src_tp = ndt::make_type<void>();
  const char *src_arrmeta = NULL;
  af->instantiate(af, copy_from_pyobject.get_type(), &ckb, 0, dt, arrmeta,
                  &src_tp, &src_arrmeta, kernel_request_single, ectx,
                  nd::array(), std::map<nd::string, ndt::type>());
  ckb(data, reinterpret_cast<char *>(&value));
  return;
}

void pydynd::array_broadcast_assign_from_py(const dynd::nd::array &a,
                                            PyObject *value,
                                            const eval::eval_context *ectx)
{
  array_broadcast_assign_from_py(a.get_type(), a.get_arrmeta(),
                                 a.get_readwrite_originptr(), value, ectx);
}

void pydynd::array_no_dim_broadcast_assign_from_py(
    const dynd::ndt::type &dt, const char *arrmeta, char *data, PyObject *value,
    const dynd::eval::eval_context *ectx)
{
  unary_ckernel_builder ckb;
  const arrfunc_type_data *af = copy_from_pyobject_no_dim_broadcast.get();
  ndt::type src_tp = ndt::make_type<void>();
  const char *src_arrmeta = NULL;
  af->instantiate(af, copy_from_pyobject_no_dim_broadcast.get_type(), &ckb, 0,
                  dt, arrmeta, &src_tp, &src_arrmeta, kernel_request_single,
                  ectx, nd::array(), std::map<nd::string, ndt::type>());
  ckb(data, reinterpret_cast<char *>(&value));
  return;
}

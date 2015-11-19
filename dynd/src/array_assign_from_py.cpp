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

void pydynd::array_broadcast_assign_from_py(
    const dynd::ndt::type &dst_tp, const char *dst_arrmeta, char *dst_data,
    PyObject *value, const dynd::eval::eval_context *ectx)
{
  // TODO: This is a hack, need a proper way to pass this dst param
  dynd::nd::array tmp_dst(
      dynd::make_array_memory_block(dst_tp.get_arrmeta_size()));
  tmp_dst.get()->tp = dst_tp;
  tmp_dst.get()->flags =
      dynd::nd::read_access_flag | dynd::nd::write_access_flag;
  if (dst_tp.get_arrmeta_size() > 0) {
    dst_tp.extended()->arrmeta_copy_construct(
        tmp_dst.get()->metadata(), dst_arrmeta,
        dynd::intrusive_ptr<dynd::memory_block_data>());
  }
  tmp_dst.get()->data = dst_data;
  dynd::ndt::type src_tp = dynd::ndt::type::make<void>();
  const char *src_arrmeta = NULL;
  char *src_data = reinterpret_cast<char *>(&value);
  const char *kwd_names[1] = {"broadcast"};
  dynd::nd::array kwd_values[1] = {true};
  (*pydynd::nd::copy_from_pyobject.get().get())(
      tmp_dst.get_type(), tmp_dst.get()->metadata(), tmp_dst.data(), 1, &src_tp,
      &src_arrmeta, &src_data, 1, kwd_values,
      std::map<std::string, dynd::ndt::type>());

  tmp_dst.get()->tp = dynd::ndt::type();
}

void
pydynd::array_broadcast_assign_from_py(const dynd::nd::array &a,
                                       PyObject *value,
                                       const dynd::eval::eval_context *ectx)
{
  array_broadcast_assign_from_py(a.get_type(), a.get()->metadata(), a.data(),
                                 value, ectx);
}

void pydynd::array_no_dim_broadcast_assign_from_py(
    const dynd::ndt::type &dst_tp, const char *dst_arrmeta, char *dst_data,
    PyObject *value, const dynd::eval::eval_context *ectx)
{
  // TODO: This is a hack, need a proper way to pass this dst param
  dynd::nd::array tmp_dst(
      dynd::make_array_memory_block(dst_tp.get_arrmeta_size()));
  tmp_dst.get()->tp = dst_tp;
  tmp_dst.get()->flags =
      dynd::nd::read_access_flag | dynd::nd::write_access_flag;
  if (dst_tp.get_arrmeta_size() > 0) {
    dst_tp.extended()->arrmeta_copy_construct(
        tmp_dst.get()->metadata(), dst_arrmeta,
        dynd::intrusive_ptr<dynd::memory_block_data>());
  }
  tmp_dst.get()->data = dst_data;
  dynd::ndt::type src_tp = dynd::ndt::type::make<void>();
  const char *src_arrmeta = NULL;
  char *src_data = reinterpret_cast<char *>(&value);
  const char *kwd_names[1] = {"broadcast"};
  dynd::nd::array kwd_values[1] = {false};
  (*pydynd::nd::copy_from_pyobject.get().get())(
      tmp_dst.get_type(), tmp_dst.get()->metadata(), tmp_dst.data(), 1, &src_tp,
      &src_arrmeta, &src_data, 1, kwd_values,
      std::map<std::string, dynd::ndt::type>());

  tmp_dst.get()->tp = dynd::ndt::type();
}

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

void pydynd::array_broadcast_assign_from_py(const dynd::ndt::type &dst_tp,
                                            const char *dst_arrmeta,
                                            char *dst_data, PyObject *value,
                                            const eval::eval_context *ectx)
{
  // TODO: This is a hack, need a proper way to pass this dst param
  nd::array tmp_dst(dynd::make_array_memory_block(dst_tp.get_arrmeta_size()));
  tmp_dst.get_ndo()->m_type = ndt::type(dst_tp).release();
  tmp_dst.get_ndo()->m_flags = nd::read_access_flag | nd::write_access_flag;
  if (dst_tp.get_arrmeta_size() > 0) {
    dst_tp.extended()->arrmeta_copy_construct(tmp_dst.get_arrmeta(),
                                              dst_arrmeta, NULL);
  }
  tmp_dst.get_ndo()->m_data_pointer = dst_data;
  ndt::type src_tp = ndt::make_type<void>();
  const char *src_arrmeta = NULL;
  char *src_data = reinterpret_cast<char *>(&value);
  copy_from_pyobject(1, &src_tp, &src_arrmeta, &src_data, kwds("dst", tmp_dst));
}

void pydynd::array_broadcast_assign_from_py(const dynd::nd::array &a,
                                            PyObject *value,
                                            const eval::eval_context *ectx)
{
  array_broadcast_assign_from_py(a.get_type(), a.get_arrmeta(),
                                 a.get_readwrite_originptr(), value, ectx);
}

void pydynd::array_no_dim_broadcast_assign_from_py(
    const dynd::ndt::type &dst_tp, const char *dst_arrmeta, char *dst_data,
    PyObject *value, const dynd::eval::eval_context *ectx)
{
  // TODO: This is a hack, need a proper way to pass this dst param
  nd::array tmp_dst(dynd::make_array_memory_block(dst_tp.get_arrmeta_size()));
  tmp_dst.get_ndo()->m_type = ndt::type(dst_tp).release();
  tmp_dst.get_ndo()->m_flags = nd::read_access_flag | nd::write_access_flag;
  if (dst_tp.get_arrmeta_size() > 0) {
    dst_tp.extended()->arrmeta_copy_construct(tmp_dst.get_arrmeta(),
                                              dst_arrmeta, NULL);
  }
  tmp_dst.get_ndo()->m_data_pointer = dst_data;
  ndt::type src_tp = ndt::make_type<void>();
  const char *src_arrmeta = NULL;
  char *src_data = reinterpret_cast<char *>(&value);
  copy_from_pyobject_no_dim_broadcast(1, &src_tp, &src_arrmeta, &src_data,
                                      kwds("dst", tmp_dst));
}

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
  pyobject_ownref result;

  // TODO: This is a hack, need a proper way to pass this dst param
  ndt::type dst_tp = ndt::make_type<void>();
  nd::array tmp_dst(dynd::make_array_memory_block(dst_tp.get_arrmeta_size()));
  tmp_dst.get_ndo()->m_type = ndt::type(dst_tp).release();
  tmp_dst.get_ndo()->m_flags = nd::read_access_flag | nd::write_access_flag;
  tmp_dst.get_ndo()->m_data_pointer =
      reinterpret_cast<char *>(result.obj_addr());
  const char *src_arrmeta = a.get_arrmeta();
  char *src_data_nonconst = const_cast<char *>(a.get_readonly_originptr());
  nd::copy_to_pyobject(1, &a.get_type(), &src_arrmeta, &src_data_nonconst,
                       kwds("dst", tmp_dst));
  if (PyErr_Occurred()) {
    throw exception();
  }
  return result.release();
}

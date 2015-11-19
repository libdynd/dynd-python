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

PyObject *pydynd::array_as_py(const dynd::nd::array &a, bool struct_as_pytuple)
{
  pyobject_ownref result;

  // TODO: This is a hack, need a proper way to pass this dst param
  dynd::ndt::type dst_tp = dynd::ndt::type::make<void>();
  dynd::nd::array tmp_dst(
      dynd::make_array_memory_block(dst_tp.get_arrmeta_size()));
  tmp_dst.get()->tp = dst_tp;
  tmp_dst.get()->flags =
      dynd::nd::read_access_flag | dynd::nd::write_access_flag;
  tmp_dst.get()->data = reinterpret_cast<char *>(result.obj_addr());
  const char *src_arrmeta = a.get()->metadata();
  char *src_data_nonconst = const_cast<char *>(a.cdata());
  (*nd::copy_to_pyobject.get().get())(dst_tp, tmp_dst.get()->metadata(),
                                      tmp_dst.data(), 1, &a.get_type(),
                                      &src_arrmeta, &src_data_nonconst, 0, NULL,
                                      std::map<std::string, dynd::ndt::type>());
  if (PyErr_Occurred()) {
    throw exception();
  }
  return result.release();
}

//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

using namespace dynd;

namespace detail {

template <type_id_t Arg0ID, type_id_t Arg0BaseID>
struct assign_to_pyobject_kernel;

template <>
struct assign_to_pyobject_kernel<bool_type_id, bool_kind_type_id>
    : nd::base_kernel<
          assign_to_pyobject_kernel<bool_type_id, bool_kind_type_id>, 1> {
  void single(char *dst, char *const *src)
  {
    Py_XDECREF(*reinterpret_cast<PyObject **>(dst));
    *reinterpret_cast<PyObject **>(dst) =
        *reinterpret_cast<bool *>(src[0]) ? Py_True : Py_False;
    Py_INCREF(*reinterpret_cast<PyObject **>(dst));
  }
};

} // namespace detail

template <type_id_t Arg0ID>
using assign_to_pyobject_kernel =
    ::detail::assign_to_pyobject_kernel<Arg0ID, base_type_id_of<Arg0ID>::value>;

namespace dynd {
namespace ndt {

  template <type_id_t Arg0ID>
  struct traits<assign_to_pyobject_kernel<Arg0ID>> {
    static type equivalent()
    {
      return callable_type::make(ndt::make_type<pyobject_type>(),
                                 {type(Arg0ID)});
    }
  };

} // namespace dynd::ndt
} // namespace dynd

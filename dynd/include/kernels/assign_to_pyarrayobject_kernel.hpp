//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

#include <numpy/arrayobject.h>

#include "types/pyobject_type.hpp"

template <type_id_t Arg0ID>
struct assign_to_pyarrayobject_kernel
    : nd::base_kernel<assign_to_pyarrayobject_kernel<Arg0ID>, 1> {
  void single(char *res, char *const *args)
  {
    std::cout << "assign_to_pyarrayobject_kernel::single" << std::endl;
    std::exit(-1);
  }
};

struct init_pyarrayobject_kernel {

};

namespace dynd {
namespace ndt {

  template <type_id_t Arg0ID>
  struct traits<assign_to_pyarrayobject_kernel<Arg0ID>> {
    static type equivalent() { return type("(Any) -> void"); }
  };

} // namespace dynd::ndt
} // namespace dynd

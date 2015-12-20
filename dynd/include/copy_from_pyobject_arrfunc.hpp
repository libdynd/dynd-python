//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <Python.h>

#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/callable.hpp>

namespace pydynd {
namespace nd {

  extern struct copy_from_pyobject : dynd::nd::declfunc<copy_from_pyobject> {
    static dynd::nd::callable make();
  } copy_from_pyobject;

} // namespace pydynd::nd
} // namespace pydynd

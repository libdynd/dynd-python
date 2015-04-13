//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <Python.h>

#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/func/arrfunc.hpp>

#include "config.hpp"

namespace pydynd {
namespace nd {

  extern struct copy_from_pyobject : declfunc<copy_from_pyobject> {
    static arrfunc children[DYND_TYPE_ID_MAX + 1];
    static arrfunc default_child;

    static arrfunc make();
  } copy_from_pyobject;

} // namespace pydynd::nd
} // namespace pydynd
//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <Python.h>

#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/func/callable.hpp>

#include "config.hpp"

namespace pydynd {
namespace nd {

  extern struct copy_from_pyobject : dynd::nd::declfunc<copy_from_pyobject> {
    static dynd::nd::callable children[DYND_TYPE_ID_MAX + 1];
    static dynd::nd::callable default_child;

    static dynd::nd::callable &overload(const dynd::ndt::type &dst_tp)
    {
      return children[dst_tp.get_type_id()];
    }

    static dynd::nd::callable make();
  } copy_from_pyobject;

} // namespace pydynd::nd
} // namespace pydynd
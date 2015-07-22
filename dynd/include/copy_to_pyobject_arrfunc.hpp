//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/arrfunc.hpp>

#include "config.hpp"

namespace pydynd {
namespace nd {

  extern struct copy_to_pyobject : dynd::nd::declfunc<copy_to_pyobject> {
    static dynd::nd::callable children[DYND_TYPE_ID_MAX + 1];
    static dynd::nd::callable default_child;

    static dynd::nd::callable make();
  } copy_to_pyobject;

} // namespace pydynd::nd
} // namespace pydynd
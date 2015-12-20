//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/callable.hpp>

namespace pydynd {
namespace nd {

  extern struct copy_to_pyobject : dynd::nd::declfunc<copy_to_pyobject> {
    static dynd::nd::callable make();
  } copy_to_pyobject;

} // namespace pydynd::nd
} // namespace pydynd

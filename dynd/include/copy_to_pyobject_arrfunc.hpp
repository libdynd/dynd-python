//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/arrfunc.hpp>

#include "config.hpp"

#define DYND_TYPE_ID_MAX 99

namespace pydynd {
namespace nd {

  extern struct copy_to_pyobject : declfunc<copy_to_pyobject> {
    static arrfunc children[DYND_TYPE_ID_MAX + 1];

    static arrfunc make();
  } copy_to_pyobject;

} // namespace pydynd::nd
} // namespace pydynd
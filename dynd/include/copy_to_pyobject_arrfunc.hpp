//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/func/arrfunc.hpp>

#include "config.hpp"

namespace pydynd {
namespace nd {

  extern struct copy_to_pyobject_dict : declfunc<copy_to_pyobject_dict> {
    static arrfunc make();
  } copy_to_pyobject_dict;

  extern struct copy_to_pyobject_tuple : declfunc<copy_to_pyobject_tuple> {
    static arrfunc make();
  } copy_to_pyobject_tuple;

} // namespace pydynd::nd
} // namespace pydynd
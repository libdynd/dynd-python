//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef PYDYND_COPY_TO_PYOBJECT_ARRFUNC_HPP
#define PYDYND_COPY_TO_PYOBJECT_ARRFUNC_HPP

#include <Python.h>

#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/func/arrfunc.hpp>

namespace pydynd {

  extern struct copy_to_pyobject_dict
      : dynd::nd::declfunc<copy_to_pyobject_dict> {
    static dynd::nd::arrfunc make();
  } copy_to_pyobject_dict;

  extern struct copy_to_pyobject_tuple
      : dynd::nd::declfunc<copy_to_pyobject_tuple> {
    static dynd::nd::arrfunc make();
  } copy_to_pyobject_tuple;

void init_copy_to_pyobject();
void cleanup_copy_to_pyobject();

} // namespace pydynd

#endif // PYDYND_COPY_TO_PYOBJECT_ARRFUNC_HPP

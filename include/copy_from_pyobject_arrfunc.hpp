//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef PYDYND_COPY_FROM_PYOBJECT_ARRFUNC_HPP
#define PYDYND_COPY_FROM_PYOBJECT_ARRFUNC_HPP

#include <Python.h>

#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/func/arrfunc.hpp>

namespace pydynd {

extern dynd::nd::arrfunc copy_from_pyobject;

extern dynd::nd::arrfunc copy_from_pyobject_no_dim_broadcast;

} // namespace pydynd

#endif // PYDYND_COPY_FROM_PYOBJECT_ARRFUNC_HPP

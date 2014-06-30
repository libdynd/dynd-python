//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef PYDYND_COPY_TO_PYOBJECT_ARRFUNC_HPP
#define PYDYND_COPY_TO_PYOBJECT_ARRFUNC_HPP

#include <Python.h>

#include <dynd/kernels/ckernel_builder.hpp>
#include <dynd/func/arrfunc.hpp>

namespace pydynd {

extern dynd::nd::arrfunc copy_to_pyobject_dict;
extern dynd::nd::arrfunc copy_to_pyobject_tuple;

} // namespace pydynd

#endif // PYDYND_COPY_TO_PYOBJECT_ARRFUNC_HPP

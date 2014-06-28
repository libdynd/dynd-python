//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef PYDYND_ARRAY_ASSIGN_TO_NUMPY_HPP
#define PYDYND_ARRAY_ASSIGN_TO_NUMPY_HPP

#include <Python.h>

#include <dynd/func/arrfunc.hpp>

namespace pydynd {

extern dynd::nd::arrfunc copy_to_numpy;

} // namespace pydynd

#endif // PYDYND_ARRAY_ASSIGN_TO_NUMPY_HPP

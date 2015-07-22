//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__EXCEPTION_TRANSLATION_HPP_
#define _DYND__EXCEPTION_TRANSLATION_HPP_

#include <Python.h>

#include "config.hpp"

namespace pydynd {

PYDYND_API void translate_exception();

PYDYND_API void set_broadcast_exception(PyObject *e);

} // namespace pydynd

#endif // _DYND__EXCEPTION_TRANSLATION_HPP_

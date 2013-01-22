//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__EXCEPTION_TRANSLATION_HPP_
#define _DYND__EXCEPTION_TRANSLATION_HPP_

#include <Python.h>

namespace pydynd {

void translate_exception();

void set_broadcast_exception(PyObject *e);

} // namespace pydynd

#endif // _DYND__EXCEPTION_TRANSLATION_HPP_

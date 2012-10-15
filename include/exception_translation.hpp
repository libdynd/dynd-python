//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__EXCEPTION_TRANSLATION_HPP_
#define _DND__EXCEPTION_TRANSLATION_HPP_

#include <Python.h>

namespace pydynd {

void translate_exception();

void set_broadcast_exception(PyObject *e);

} // namespace pydynd

#endif // _DND__EXCEPTION_TRANSLATION_HPP_

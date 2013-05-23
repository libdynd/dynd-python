//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__NDARRAY_AS_PY_HPP_
#define _DYND__NDARRAY_AS_PY_HPP_

#include <Python.h>

#include <dynd/ndobject.hpp>

namespace pydynd {

/**
 * Converts an ndobject into a Python object
 * using the default settings.
 */
PyObject *ndobject_as_py(const dynd::ndobject& n);

} // namespace pydynd

#endif // _DYND__NDARRAY_AS_PY_HPP_

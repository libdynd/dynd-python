//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__NDARRAY_FROM_PY_HPP_
#define _DYND__NDARRAY_FROM_PY_HPP_

#include <Python.h>

#include <dynd/ndobject.hpp>

namespace pydynd {

/**
 * Converts a Python object into an ndobject using
 * the default settings.
 */
dynd::ndobject ndobject_from_py(PyObject *obj);

} // namespace pydynd

#endif // _DYND__NDARRAY_FROM_PY_HPP_


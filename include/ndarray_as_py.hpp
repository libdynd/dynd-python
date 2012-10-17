//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__NDARRAY_AS_PY_HPP_
#define _DYND__NDARRAY_AS_PY_HPP_

#include <Python.h>

#include <dynd/ndarray.hpp>

namespace pydynd {

/**
 * Converts an ndarray into a Python object
 * using the default settings.
 */
PyObject *ndarray_as_py(const dynd::ndarray& n);

} // namespace pydynd

#endif // _DYND__NDARRAY_AS_PY_HPP_

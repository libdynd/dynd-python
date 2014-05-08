//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__NDARRAY_AS_PY_HPP_
#define _DYND__NDARRAY_AS_PY_HPP_

#include <Python.h>

#include <dynd/array.hpp>

namespace pydynd {

/**
 * Converts an nd::array into a Python object
 * using the default settings.
 *
 * \param n  The nd::array to convert into a PyObject*.
 * \param struct_as_pytuple  If true, converts structs into tuples, otherwise
 *                           converts them into dicts.
 */
PyObject *array_as_py(const dynd::nd::array& n, bool struct_as_pytuple);

} // namespace pydynd

#endif // _DYND__NDARRAY_AS_PY_HPP_

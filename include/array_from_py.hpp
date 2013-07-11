//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__NDOBJECT_FROM_PY_HPP_
#define _DYND__NDOBJECT_FROM_PY_HPP_

#include <Python.h>

#include <dynd/array.hpp>

namespace pydynd {

/**
 * Converts a Python object into an nd::array using
 * the default settings. This function automatically
 * detects the type to use from the input Python object.
 *
 * \param obj  The PyObject to convert to an nd::array.
 */
dynd::nd::array array_from_py(PyObject *obj);

/**
 * Converts a Python object into an nd::array using
 * the default settings, using the provided type as a
 * array data type.
 *
 * \param obj  The PyObject to convert to an nd::array.
 * \param dt  The dynd type to use. Additional array dimensions
 *            may be prepended to this type.
 * \param uniform  If True, then 'dt' must be a zero-dimensional type,
 *                 and the shape is automatically deduced. If False,
 *                 then 'dt' must already contain all the array dimensions.
 */
dynd::nd::array array_from_py(PyObject *obj, const dynd::ndt::type& dt, bool uniform);

} // namespace pydynd

#endif // _DYND__NDOBJECT_FROM_PY_HPP_


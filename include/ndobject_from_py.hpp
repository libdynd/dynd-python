//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__NDOBJECT_FROM_PY_HPP_
#define _DYND__NDOBJECT_FROM_PY_HPP_

#include <Python.h>

#include <dynd/ndobject.hpp>

namespace pydynd {

/**
 * Converts a Python object into an ndobject using
 * the default settings. This function automatically
 * detects the dtype to use from the input Python object.
 *
 * \param obj  The PyObject to convert to an ndobject.
 */
dynd::ndobject ndobject_from_py(PyObject *obj);

/**
 * Converts a Python object into an ndobject using
 * the default settings, using the provided dtype as a
 * uniform dtype.
 *
 * \param obj  The PyObject to convert to an ndobject.
 * \param dt  The dtype to use. Additional uniform dimensions
 *            may be prepended to this dtype.
 * \param uniform  If True, then 'dt' must be a zero-dimensional dtype,
 *                 and the shape is automatically deduced. If False,
 *                 then 'dt' must already contain all the uniform dimensions.
 */
dynd::ndobject ndobject_from_py(PyObject *obj, const dynd::dtype& dt, bool uniform);

} // namespace pydynd

#endif // _DYND__NDOBJECT_FROM_PY_HPP_


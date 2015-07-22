//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__ARRAY_FROM_PY_HPP_
#define _DYND__ARRAY_FROM_PY_HPP_

#include <Python.h>

#include <dynd/array.hpp>

#include "config.hpp"

namespace pydynd {

/**
 * Converts a Python object into an nd::array using
 * the default settings. This function automatically
 * detects the type to use from the input Python object.
 *
 * \param obj  The PyObject to convert to an nd::array.
 * \param access_flags Either 0 to inherit the object's access
 *                     flags, or the access flags for the result.
 *                     The default if no access was specified
 *                     is immutable.
 * \param always_copy If this is set to true, a new copy is always
 *                    created.
 */
dynd::nd::array array_from_py(PyObject *obj, uint32_t access_flags,
                              bool always_copy,
                              const dynd::eval::eval_context *ectx);

/**
 * Converts a Python object into an nd::array using
 * the default settings, using the provided type as a
 * array data type.
 *
 * This function always creates a new array, it never
 * creates a view into an existing array.
 *
 * \param obj  The PyObject to convert to an nd::array.
 * \param tp  The dynd type to use. Additional array dimensions
 *            may be prepended to this type if ``fulltype`` is not specified.
 * \param fulltype  If True, then ``tp`` contains the full type of the
 *                  resulting array. If False, then
 *                  ``tp`` is a dtype which should be the trailing type of
 *                  the result, and the function analyzes the object
 *                  to deduce any additional leading dimensions.
 * \param access_flags The access flags for the result, or 0 for the default
 *                     immutable.
 */
dynd::nd::array array_from_py(PyObject *obj, const dynd::ndt::type &tp,
                              bool fulltype, uint32_t access_flags,
                              const dynd::eval::eval_context *ectx);

void init_array_from_py();

} // namespace pydynd

#endif // _DYND__ARRAY_FROM_PY_HPP_

//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__ARRAY_FROM_PY_HPP_
#define _DYND__ARRAY_FROM_PY_HPP_

#include <Python.h>

#include <dynd/types/type_id.hpp>

#include "visibility.hpp"

#include <dynd/array.hpp>

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
PYDYND_API dynd::nd::array array_from_py(PyObject *obj, uint32_t access_flags,
                                         bool always_copy);

PYDYND_API dynd::ndt::type xtype_for_prefix(PyObject *obj);


void init_array_from_py();

} // namespace pydynd

#endif // _DYND__ARRAY_FROM_PY_HPP_

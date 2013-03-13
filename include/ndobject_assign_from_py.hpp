//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__NDOBJECT_ASSIGN_FROM_PY_HPP_
#define _DYND__NDOBJECT_ASSIGN_FROM_PY_HPP_

#include <Python.h>

#include <dynd/ndobject.hpp>

namespace pydynd {

/**
 * Assigns the values from 'obj' to 'n', broadcasting
 * the input if requested.
 *
 * \param n  The ndobject which is being assigned to.
 * \param obj  value PyObject for the source data.
 */
void ndobject_broadcast_assign_from_py(const dynd::ndobject& n, PyObject *value);

/**
 * Assigns the values from 'obj' to the 'dt/metadata/data' raw ndobject, broadcasting
 * the input if requested.
 *
 * \param dt  The dtype of the destination.
 * \param metadata  The metadata of the destination.
 * \param data  The data of the destination.
 * \param value The PyObject for the source data.
 */
void ndobject_broadcast_assign_from_py(const dynd::dtype& dt, const char *metadata, char *data, PyObject *value);

} // namespace pydynd

#endif // _DYND__NDOBJECT_ASSIGN_FROM_PY_HPP_


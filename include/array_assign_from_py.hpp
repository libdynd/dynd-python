//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__NDOBJECT_ASSIGN_FROM_PY_HPP_
#define _DYND__NDOBJECT_ASSIGN_FROM_PY_HPP_

#include <Python.h>

#include <dynd/array.hpp>

namespace pydynd {

/**
 * Assigns the values from 'obj' to 'n', broadcasting
 * the input if requested.
 *
 * \param n  The array which is being assigned to.
 * \param obj  value PyObject for the source data.
 */
void array_broadcast_assign_from_py(const dynd::nd::array& n, PyObject *value);

/**
 * Assigns the values from 'obj' to the 'dt/metadata/data' raw nd::array, broadcasting
 * the input.
 *
 * \param dt  The dynd type of the destination.
 * \param metadata  The metadata of the destination.
 * \param data  The data of the destination.
 * \param value The PyObject for the source data.
 */
void array_broadcast_assign_from_py(const dynd::ndt::type& dt, const char *metadata, char *data, PyObject *value);

/**
 * Assigns the values from 'obj' to the 'dt/metadata/data' raw nd::array, broadcasting
 * the individual input dimensions, but not broadcasting by skipping dimensions.
 *
 * \param dt  The dynd type of the destination.
 * \param metadata  The metadata of the destination.
 * \param data  The data of the destination.
 * \param value The PyObject for the source data.
 */
void array_nodim_broadcast_assign_from_py(const dynd::ndt::type& dt, const char *metadata, char *data, PyObject *value);

} // namespace pydynd

#endif // _DYND__NDOBJECT_ASSIGN_FROM_PY_HPP_


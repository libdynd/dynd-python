//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef PYDYND_ARRAY_ASSIGN_FROM_PY_HPP
#define PYDYND_ARRAY_ASSIGN_FROM_PY_HPP

#include <Python.h>

#include <dynd/array.hpp>

#include "config.hpp"

namespace pydynd {

/**
 * Assigns the values from 'obj' to 'a', broadcasting
 * the input if requested.
 *
 * \param a  The array which is being assigned to.
 * \param value  value PyObject for the source data.
 */
void array_broadcast_assign_from_py(const dynd::nd::array &a, PyObject *value,
                                    const dynd::eval::eval_context *ectx);

/**
 * Assigns the values from 'obj' to the 'dt/arrmeta/data' raw nd::array,
 * broadcasting the input.
 *
 * \param dt  The dynd type of the destination.
 * \param arrmeta  The arrmeta of the destination.
 * \param data  The data of the destination.
 * \param value The PyObject for the source data.
 */
void array_broadcast_assign_from_py(const dynd::ndt::type &dt,
                                    const char *arrmeta, char *data,
                                    PyObject *value,
                                    const dynd::eval::eval_context *ectx);

/**
 * Assigns the values from 'obj' to the 'dt/arrmeta/data' raw nd::array,
 * broadcasting the individual input dimensions, but not broadcasting
 * by skipping dimensions.
 *
 * \param dt  The dynd type of the destination.
 * \param arrmeta  The arrmeta of the destination.
 * \param data  The data of the destination.
 * \param value The PyObject for the source data.
 */
void array_no_dim_broadcast_assign_from_py(
    const dynd::ndt::type &dt, const char *arrmeta, char *data, PyObject *value,
    const dynd::eval::eval_context *ectx);

} // namespace pydynd

#endif // PYDYND_ARRAY_ASSIGN_FROM_PY_HPP

//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef PYDYND_ARRAY_ASSIGN_FROM_PY_HPP
#define PYDYND_ARRAY_ASSIGN_FROM_PY_HPP

#include <Python.h>

#include <dynd/array.hpp>

#include "visibility.hpp"

namespace pydynd {

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
                                    PyObject *value);

} // namespace pydynd

#endif // PYDYND_ARRAY_ASSIGN_FROM_PY_HPP

//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__NDARRAY_AS_PEP3118_HPP_
#define _DYND__NDARRAY_AS_PEP3118_HPP_

#include <Python.h>

#include <dynd/array.hpp>

#include "config.hpp"

namespace pydynd {

/**
 * \brief Converts a dynd type into a PEP 3118 format string.
 *
 * \param dt  The dynd type to convert.
 * \param arrmeta  If non-NULL, arrmeta to provide additional strides/offsets
 *                 not available in just the type.
 *
 * \returns  A PEP3118 format string.
 */
std::string make_pep3118_format(intptr_t &out_itemsize,
                                const dynd::ndt::type &dt,
                                const char *arrmeta = NULL);

/**
 * \brief Converts an nd::array into a PEP3118 buffer.
 */
PYDYND_API int array_getbuffer_pep3118(PyObject *ndo, Py_buffer *buffer,
                                       int flags);

/**
 * \brief Frees a previously created PEP3118 buffer.
 */
PYDYND_API int array_releasebuffer_pep3118(PyObject *ndo, Py_buffer *buffer);

} // namespace pydynd

#endif // _DYND__NDARRAY_AS_PEP3118_HPP_

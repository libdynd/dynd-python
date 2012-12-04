//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__NDARRAY_AS_PEP3118_HPP_
#define _DYND__NDARRAY_AS_PEP3118_HPP_

#include <Python.h>

#include <dynd/ndobject.hpp>

namespace pydynd {

/**
 * \brief Converts a dynd dtype into a PEP 3118 format string.
 *
 * \param dt  The dtype to convert.
 * \param metadata  If non-NULL, dtype metadata to provide additional strides/offsets
 *                  not available in just the dtype.
 *
 * \returns  A PEP3118 format string.
 */
std::string make_pep3118_format(intptr_t& out_itemsize, const dynd::dtype& dt, const char *metadata = NULL);

/**
 * \brief Converts an ndobject into a PEP3118 buffer.
 */
int ndobject_getbuffer_pep3118(PyObject *ndo, Py_buffer *buffer, int flags);

/**
 * \brief Frees a previously created PEP3118 buffer.
 */
int ndobject_releasebuffer_pep3118(PyObject *ndo, Py_buffer *buffer);

} // namespace pydynd

#endif // _DYND__NDARRAY_AS_PEP3118_HPP_

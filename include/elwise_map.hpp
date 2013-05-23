//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__ELWISE_MAP_HPP_
#define _DYND__ELWISE_MAP_HPP_

#include <Python.h>

#include <dynd/ndobject.hpp>

namespace pydynd {

/**
 * Applies a Python callable of a specific form to every
 * element of the provided ndobject.
 *
 * The Python callable should have the following structure.
 * It is called with two corresponding one-dimensional ndobjects,
 * with 'dst_type' and 'src_type' respectively. The code must
 * not retain any references to the ndobjects, as they are temporary,
 * reused shells just for providing the data to the function.
 *
 *      def my_mapper(dst, src):
 *          dst[...] = some_operaton(src)
 *
 * \param n_obj  A WNDObject containing the ndobject to wrap.
 * \param callable  The Python callable which does the mapping.
 * \param dst_type  A dynd type for the destination elements.
 * \param src_type  A dynd type for the source elements.
 */
PyObject *elwise_map(PyObject *n_obj, PyObject *callable, PyObject *dst_type, PyObject *src_type);

} // namespace pydynd

#endif // _DYND__ELWISE_MAP_HPP_

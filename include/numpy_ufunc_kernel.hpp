//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__NUMPY_UFUNC_KERNEL_HPP_
#define _DYND__NUMPY_UFUNC_KERNEL_HPP_

#include "numpy_interop.hpp"
#include <numpy/ufuncobject.h>

namespace pydynd {

/**
 * Returns a list of type tuples, one for each loop
 * registered in the ufunc, either in the main functions
 * list or the userloops.
 *
 * NOTE: For ufuncs that customize the type resolution,
 *       such as the datetime64 type, this may not capture
 *       all of the loops, as the customization can arbitrarily
 *       introduce/create more loops.
 *
 * \param ufunc  The numpy ufunc to analyze. This should be a
 *               PyUFuncObject*, but its type is checked.
 *
 * \returns NULL on error (with a Python exception set),
 *          or a list of type tuples.
 */
PyObject *numpy_typetuples_from_ufunc(PyObject *ufunc);

} // namespace pydynd

#endif // _DYND__NUMPY_UFUNC_KERNEL_HPP_
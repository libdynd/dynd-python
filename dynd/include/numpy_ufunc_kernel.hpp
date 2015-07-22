//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include "config.hpp"
#include "numpy_interop.hpp"
#include <numpy/ufuncobject.h>
#include <dynd/func/callable.hpp>

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

namespace nd {
  namespace functional {

    /**
     * Given a ufunc and a type tuple supported by that ufunc,
     * returns an callable.
     *
     * \param ufunc  The numpy ufunc.
     * \param type_tuple  The tuple of types defining the kernel signature.
     * \param ckernel_acquires_gil  A flag which controls whether the ckernel
     *                              should acquire the GIL before calling the
     *                              ufunc's kernel. Since numpy doesn't have a
     *                              nice, rigorous way of specifying the need
     *                              for this, we make it a parameter.
     *
     * \returns An callable inside an nd::array.
     */
    PyObject *callable_from_ufunc(PyObject *ufunc, PyObject *type_tuple,
                                  int ckernel_acquires_gil);

  } // namespace pydynd::nd::functional
} // namespace pydynd::nd
} // namespace pydynd

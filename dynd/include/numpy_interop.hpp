//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// This header defines some functions to
// interoperate with numpy
//

#ifndef _DYND__NUMPY_INTEROP_HPP_
#define _DYND__NUMPY_INTEROP_HPP_

#include <dynd/array.hpp>

#include "numpy_type_interop.hpp"
#if DYND_NUMPY_INTEROP

#include "type_functions.hpp"
#include "utility_functions.hpp"
#include "visibility.hpp"

namespace pydynd {

/**
 * When the function _type_from_numpy_dtype returns a type which requires
 * additional arrmeta to be filled in, this function should be called to
 * populate that arrmeta in a created nd::array.
 *
 * \param tp  The dynd type returned by _type_from_numpy_dtype.
 * \param d  The numpy dtype passed to _type_from_numpy_dtype.
 * \param arrmeta  A pointer to the arrmeta to populate.
 */
void fill_arrmeta_from_numpy_dtype(const dynd::ndt::type &tp, PyArray_Descr *d, char *arrmeta);

/**
 * Views or copies a numpy PyArrayObject as an nd::array.
 *
 * \param obj  The numpy array object.
 * \param access_flags  The requested access flags (0 for default).
 * \param always_copy  If true, produce a copy instead of a view.
 */
dynd::nd::array PYDYND_API array_from_numpy_array(PyArrayObject *obj, uint32_t access_flags, bool always_copy);

// Convenience wrapper for use in Cython where the type has already been
// checked.
inline dynd::nd::array array_from_numpy_array_cast(PyObject *obj, uint32_t access_flags, bool always_copy)
{
  return array_from_numpy_array(reinterpret_cast<PyArrayObject *>(obj), access_flags, always_copy);
}

/**
 * Creates a dynd::nd::array from a numpy scalar. This always produces
 * a copy.
 *
 * \param obj  The numpy scalar object.
 * \param access_flags  The requested access flags (0 for default).
 */
dynd::nd::array array_from_numpy_scalar(PyObject *obj, uint32_t access_flags);

} // namespace pydynd

#endif // DYND_NUMPY_INTEROP

#endif // _DYND__NUMPY_INTEROP_HPP_

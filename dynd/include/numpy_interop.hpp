//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// This header defines some functions to
// interoperate with numpy
//

#ifndef _DYND__NUMPY_INTEROP_HPP_
#define _DYND__NUMPY_INTEROP_HPP_

#include <Python.h>

// Define this to 1 or 0 depending on whether numpy interop
// should be compiled in.
#define DYND_NUMPY_INTEROP 1

// Only expose the things in this header when numpy interop is enabled
#if DYND_NUMPY_INTEROP

#include <numpy/numpyconfig.h>

// Don't use the deprecated Numpy functions
#ifdef NPY_1_7_API_VERSION
#define NPY_NO_DEPRECATED_API 8 // NPY_1_7_API_VERSION
#else
#define NPY_ARRAY_NOTSWAPPED NPY_NOTSWAPPED
#define NPY_ARRAY_ALIGNED NPY_ALIGNED
#define NPY_ARRAY_WRITEABLE NPY_WRITEABLE
#define NPY_ARRAY_UPDATEIFCOPY NPY_UPDATEIFCOPY
#endif

#define PY_ARRAY_UNIQUE_SYMBOL pydynd_ARRAY_API
#define PY_UFUNC_UNIQUE_SYMBOL pydynd_UFUNC_API
// Invert the importing signal to match how numpy wants it
#ifndef NUMPY_IMPORT_ARRAY
#define NO_IMPORT_ARRAY
#define NO_IMPORT_UFUNC
#endif

#include <sstream>

#include <dynd/type.hpp>
#include <dynd/array.hpp>

#include <numpy/ndarrayobject.h>
#include <numpy/ufuncobject.h>

#include "config.hpp"

#ifndef NPY_DATETIME_NAT
#define NPY_DATETIME_NAT NPY_MIN_INT64
#endif
#ifndef NPY_ARRAY_WRITEABLE
#define NPY_ARRAY_WRITEABLE NPY_WRITEABLE
#endif
#ifndef NPY_ARRAY_ALIGNED
#define NPY_ARRAY_ALIGNED NPY_ALIGNED
#endif

namespace pydynd {

inline int import_numpy()
{
#ifdef NUMPY_IMPORT_ARRAY
  import_array1(-1);
  import_umath1(-1);
#endif

  return 0;
}

/**
 * Converts a numpy dtype to a dynd type. Use the data_alignment
 * parameter to get accurate alignment, as Numpy may have misaligned data,
 * or may report a smaller alignment than is necessary based on the data.
 *
 * \param d  The numpy dtype to convert.
 * \param data_alignment  If associated with particular data, the actual
 *                        alignment of that data. The default of zero
 *                        causes it to use Numpy's data alignment.
 *
 * \returns  The dynd equivalent of the numpy dtype.
 */
dynd::ndt::type _type_from_numpy_dtype(PyArray_Descr *d,
                                          size_t data_alignment = 0);

/**
 * Converts a numpy type number to a dynd type. This produces an
 * aligned output type.
 *
 * \param numpy_type_num  The numpy type number.
 *
 * \returns  The dynd equivalent of the numpy dtype.
 */
dynd::ndt::type _type_from_numpy_type_num(int numpy_type_num);

/**
 * Given a NumPy struct/record dtype, extracts the field types, names,
 * and offsets.
 *
 * \param d  The struct/record dtype.
 * \param out_field_dtypes  This is filled with borrowed references to the field
 *                          dtypes.
 * \param out_field_names  This is filled with the field names.
 * \param out_field_offsets  This is filled with the field offsets.
 */
void extract_fields_from_numpy_struct(
    PyArray_Descr *d, std::vector<PyArray_Descr *> &out_field_dtypes,
    std::vector<std::string> &out_field_names,
    std::vector<size_t> &out_field_offsets);

/**
 * When the function _type_from_numpy_dtype returns a type which requires
 * additional arrmeta to be filled in, this function should be called to
 * populate that arrmeta in a created nd::array.
 *
 * \param tp  The dynd type returned by _type_from_numpy_dtype.
 * \param d  The numpy dtype passed to _type_from_numpy_dtype.
 * \param arrmeta  A pointer to the arrmeta to populate.
 */
void fill_arrmeta_from_numpy_dtype(const dynd::ndt::type &tp, PyArray_Descr *d,
                                   char *arrmeta);

/**
 * Converts a dynd type to a numpy dtype.
 *
 * \param tp  The dynd type to convert.
 */
PYDYND_API PyArray_Descr *numpy_dtype_from__type(const dynd::ndt::type &tp);

/** Small helper for cython parameter */
inline PyObject *numpy_dtype_obj_from__type(const dynd::ndt::type &tp)
{
  return (PyObject *)numpy_dtype_from__type(tp);
}

/**
 * Converts a dynd type to a numpy dtype, also supporting types which
 * rely on their arrmeta for field offset information.
 *
 * \param tp  The dynd type to convert.
 * \param arrmeta  The arrmeta for the dynd type.
 */
PYDYND_API PyArray_Descr *numpy_dtype_from__type(const dynd::ndt::type &tp,
                                                 const char *arrmeta);

/**
 * Converts a pytypeobject for a n`umpy scalar
 * into a dynd type.
 *
 * Returns 0 on success, -1 if it didn't match.
 */
int _type_from_numpy_scalar_typeobject(PyTypeObject *obj,
                                          dynd::ndt::type &out_tp);

/**
 * Gets the dynd type of a numpy scalar object
 */
dynd::ndt::type _type_of_numpy_scalar(PyObject *obj);

/**
 * Views or copies a numpy PyArrayObject as an nd::array.
 *
 * \param obj  The numpy array object.
 * \param access_flags  The requested access flags (0 for default).
 * \param always_copy  If true, produce a copy instead of a view.
 */
dynd::nd::array array_from_numpy_array(PyArrayObject *obj,
                                       uint32_t access_flags, bool always_copy);

/**
 * Creates a dynd::nd::array from a numpy scalar. This always produces
 * a copy.
 *
 * \param obj  The numpy scalar object.
 * \param access_flags  The requested access flags (0 for default).
 */
dynd::nd::array array_from_numpy_scalar(PyObject *obj, uint32_t access_flags);

/**
 * Returns the numpy kind ('i', 'f', etc) of the array.
 */
char numpy_kindchar_of(const dynd::ndt::type &tp);

} // namespace pydynd

#endif // DYND_NUMPY_INTEROP

// Make a no-op import_numpy for Cython to call,
// so it doesn't need to know about DYND_NUMPY_INTEROP
#if !DYND_NUMPY_INTEROP
namespace pydynd {

inline int import_numpy() { return 0; }

// If we're not building against Numpy, define our
// own version of this struct to use.
typedef struct {
  int two; /* contains the integer 2 as a sanity check */

  int nd; /* number of dimensions */

  char typekind; /* kind in array --- character code of typestr */

  int element_size; /* size of each element */

  int flags; /* how should be data interpreted. Valid
              * flags are CONTIGUOUS (1), F_CONTIGUOUS (2),
              * ALIGNED (0x100), NOTSWAPPED (0x200), and
              * WRITEABLE (0x400).  ARR_HAS_DESCR (0x800)
              * states that arrdescr field is present in
              * structure
              */

  npy_intp *shape; /* A length-nd array of shape information */

  npy_intp *strides; /* A length-nd array of stride information */

  void *data; /* A pointer to the first element of the array */

  PyObject *descr; /* A list of fields or NULL (ignored if flags
                    * does not have ARR_HAS_DESCR flag set)
                    */
} PyArrayInterface;

} // namespace pydynd
#endif // !DYND_NUMPY_INTEROP

#endif // _DYND__NUMPY_INTEROP_HPP_

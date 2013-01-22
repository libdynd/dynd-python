//
// Copyright (C) 2011-13, DyND Developers
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
# define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#else
# define NPY_ARRAY_NOTSWAPPED   NPY_NOTSWAPPED
# define NPY_ARRAY_ALIGNED      NPY_ALIGNED
# define NPY_ARRAY_WRITEABLE    NPY_WRITEABLE
# define NPY_ARRAY_UPDATEIFCOPY NPY_UPDATEIFCOPY
#endif

#define PY_ARRAY_UNIQUE_SYMBOL pydynd_ARRAY_API
// Invert the importing signal to match how numpy wants it
#ifndef NUMPY_IMPORT_ARRAY
# define NO_IMPORT_ARRAY
#endif

#include <sstream>

#include <dynd/dtype.hpp>
#include <dynd/ndobject.hpp>

#include <numpy/ndarrayobject.h>
#include <numpy/ufuncobject.h>

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
 * Converts a numpy dtype to a dynd::dtype. Use the data_alignment
 * parameter to get accurate alignment, as Numpy may have misaligned data,
 * or may report a smaller alignment than is necessary based on the data.
 *
 * \param d  The Numpy dtype to convert.
 * \param data_alignment  If associated with particular data, the actual
 *                        alignment of that data. The default of zero
 *                        causes it to use Numpy's data alignment.
 *
 * \returns  The dynd equivalent of the Numpy dtype.
 */
dynd::dtype dtype_from_numpy_dtype(PyArray_Descr *d, size_t data_alignment = 0);

/**
 * When the function dtype_from_numpy_dtype returns a dtype which requires
 * additional metadata to be filled in, this function should be called to populate
 * that metadata in a created ndobject.
 *
 * \param dt  The dtype returned by dtype_from_numpy_dtype.
 * \param d  The numpy dtype passed to dtype_from_numpy_dtype.
 * \param metadata  A pointer to the metadata to populate.
 */
void fill_metadata_from_numpy_dtype(const dynd::dtype& dt, PyArray_Descr *d, char *metadata);

/**
 * Converts a dynd::dtype to a numpy dtype.
 *
 * \param dt  The dtype to convert.
 */
PyArray_Descr *numpy_dtype_from_dtype(const dynd::dtype& dt);

/**
 * Converts a dynd::dtype to a numpy dtype, also supporting dtypes which
 * rely on their metadata for field offset information.
 *
 * \param dt  The dtype to convert.
 * \param metadata  The metadata for the dtype.
 */
PyArray_Descr *numpy_dtype_from_dtype(const dynd::dtype& dt, const char *metadata);

/**
 * Converts a pytypeobject for a numpy scalar
 * into a dynd::dtype.
 *
 * Returns 0 on success, -1 if it didn't match.
 */
int dtype_from_numpy_scalar_typeobject(PyTypeObject* obj, dynd::dtype& out_d);

/**
 * Gets the dtype of a numpy scalar object
 */
dynd::dtype dtype_of_numpy_scalar(PyObject* obj);

/**
 * Views a Numpy PyArrayObject as a dynd::ndobject.
 */
dynd::ndobject ndobject_from_numpy_array(PyArrayObject* obj);

/**
 * Creates a dynd::ndobject from a numpy scalar.
 */
dynd::ndobject ndobject_from_numpy_scalar(PyObject* obj);

/**
 * Returns the numpy kind ('i', 'f', etc) of the array.
 */
char numpy_kindchar_of(const dynd::dtype& d);

} // namespace pydynd

#endif // DYND_NUMPY_INTEROP

// Make a no-op import_numpy for Cython to call,
// so it doesn't need to know about DYND_NUMPY_INTEROP
#if !DYND_NUMPY_INTEROP
namespace pydynd {

inline int import_numpy()
{
    return 0;
}

// If we're not building against Numpy, define our
// own version of this struct to use.
typedef struct {
    int two;              /*
                           * contains the integer 2 as a sanity
                           * check
                           */

    int nd;               /* number of dimensions */

    char typekind;        /*
                           * kind in array --- character code of
                           * typestr
                           */

    int element_size;         /* size of each element */

    int flags;            /*
                           * how should be data interpreted. Valid
                           * flags are CONTIGUOUS (1), F_CONTIGUOUS (2),
                           * ALIGNED (0x100), NOTSWAPPED (0x200), and
                           * WRITEABLE (0x400).  ARR_HAS_DESCR (0x800)
                           * states that arrdescr field is present in
                           * structure
                           */

    npy_intp *shape;       /*
                            * A length-nd array of shape
                            * information
                            */

    npy_intp *strides;    /* A length-nd array of stride information */

    void *data;           /* A pointer to the first element of the array */

    PyObject *descr;      /*
                           * A list of fields or NULL (ignored if flags
                           * does not have ARR_HAS_DESCR flag set)
                           */
} PyArrayInterface;

} // namespace pydynd
#endif // !DYND_NUMPY_INTEROP

#endif // _DYND__NUMPY_INTEROP_HPP_

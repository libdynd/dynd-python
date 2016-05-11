//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// This header defines some functions to
// interoperate with numpy
//

#pragma once

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

#include <numpy/ndarrayobject.h>
#include <numpy/ufuncobject.h>

#ifndef NPY_DATETIME_NAT
#define NPY_DATETIME_NAT NPY_MIN_INT64
#endif
#ifndef NPY_ARRAY_WRITEABLE
#define NPY_ARRAY_WRITEABLE NPY_WRITEABLE
#endif
#ifndef NPY_ARRAY_ALIGNED
#define NPY_ARRAY_ALIGNED NPY_ALIGNED
#endif

inline int import_numpy()
{
#ifdef NUMPY_IMPORT_ARRAY
  import_array1(-1);
  import_umath1(-1);
#endif

  return 0;
}

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

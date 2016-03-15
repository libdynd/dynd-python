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

#include <string>
#include <vector>

#include "utility_functions.hpp"
#include "type_functions.hpp"

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

#include <dynd/array.hpp>
#include <dynd/type.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/struct_type.hpp>

#include <numpy/ndarrayobject.h>
#include <numpy/ufuncobject.h>

#include "visibility.hpp"

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
 * Converts a numpy type number to a dynd type. This produces an
 * aligned output type.
 *
 * \param numpy_type_num  The numpy type number.
 *
 * \returns  The dynd equivalent of the numpy dtype.
 */
inline dynd::ndt::type _type_from_numpy_type_num(int numpy_type_num)
{
  switch (numpy_type_num) {
  case NPY_BOOL:
    return dynd::ndt::make_type<dynd::bool1>();
  case NPY_BYTE:
    return dynd::ndt::make_type<npy_byte>();
  case NPY_UBYTE:
    return dynd::ndt::make_type<npy_ubyte>();
  case NPY_SHORT:
    return dynd::ndt::make_type<npy_short>();
  case NPY_USHORT:
    return dynd::ndt::make_type<npy_ushort>();
  case NPY_INT:
    return dynd::ndt::make_type<npy_int>();
  case NPY_UINT:
    return dynd::ndt::make_type<npy_uint>();
  case NPY_LONG:
    return dynd::ndt::make_type<npy_long>();
  case NPY_ULONG:
    return dynd::ndt::make_type<npy_ulong>();
  case NPY_LONGLONG:
    return dynd::ndt::make_type<npy_longlong>();
  case NPY_ULONGLONG:
    return dynd::ndt::make_type<npy_ulonglong>();
#if NPY_API_VERSION >= 6 // At least NumPy 1.6
  case NPY_HALF:
    return dynd::ndt::make_type<dynd::float16>();
#endif
  case NPY_FLOAT:
    return dynd::ndt::make_type<float>();
  case NPY_DOUBLE:
    return dynd::ndt::make_type<double>();
  case NPY_CFLOAT:
    return dynd::ndt::make_type<dynd::complex<float>>();
  case NPY_CDOUBLE:
    return dynd::ndt::make_type<dynd::complex<double>>();
  default: {
    std::stringstream ss;
    ss << "Cannot convert numpy type num " << numpy_type_num
       << " to a dynd type";
    throw dynd::type_error(ss.str());
  }
  }
}

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
// This doesn't rely on any static initialization from numpy,
// so it should be fine to inline.
inline void
extract_fields_from_numpy_struct(PyArray_Descr *d,
                                 std::vector<PyArray_Descr *> &out_field_dtypes,
                                 std::vector<std::string> &out_field_names,
                                 std::vector<size_t> &out_field_offsets)
{
  if (!PyDataType_HASFIELDS(d)) {
    throw dynd::type_error(
        "Tried to treat a non-structured NumPy dtype as a structure");
  }

  PyObject *names = d->names;
  Py_ssize_t names_size = PyTuple_GET_SIZE(names);

  for (Py_ssize_t i = 0; i < names_size; ++i) {
    PyObject *key = PyTuple_GET_ITEM(names, i);
    PyObject *tup = PyDict_GetItem(d->fields, key);
    PyArray_Descr *fld_dtype;
    PyObject *title;
    int offset = 0;
    if (!PyArg_ParseTuple(tup, "Oi|O", &fld_dtype, &offset, &title)) {
      throw dynd::type_error("Numpy struct dtype has corrupt data");
    }
    out_field_dtypes.push_back(fld_dtype);
    out_field_names.push_back(pystring_as_string(key));
    out_field_offsets.push_back(offset);
  }
}

inline dynd::ndt::type _type_from_numpy_dtype(PyArray_Descr *d,
                                              size_t data_alignment = 0);

inline dynd::ndt::type make_struct_type_from_numpy_struct(PyArray_Descr *d,
                                                          size_t data_alignment)
{
  std::vector<PyArray_Descr *> field_dtypes;
  std::vector<std::string> field_names;
  std::vector<size_t> field_offsets;

  pydynd::extract_fields_from_numpy_struct(d, field_dtypes, field_names,
                                           field_offsets);

  std::vector<dynd::ndt::type> field_types;

  if (data_alignment == 0) {
    data_alignment = (size_t)d->alignment;
  }

  // The alignment must divide into the total element size,
  // shrink it until it does.
  while (!dynd::offset_is_aligned((size_t)d->elsize, data_alignment)) {
    data_alignment >>= 1;
  }

  for (size_t i = 0; i < field_dtypes.size(); ++i) {
    PyArray_Descr *fld_dtype = field_dtypes[i];
    size_t offset = field_offsets[i];
    field_types.push_back(
        pydynd::_type_from_numpy_dtype(fld_dtype, data_alignment));
    // If the field isn't aligned enough, turn it into an unaligned type
    if (!dynd::offset_is_aligned(offset | data_alignment,
                                 field_types.back().get_data_alignment())) {
      throw std::runtime_error("field isn't unaligned");
    }
  }

  // Make a cstruct if possible, struct otherwise
  return dynd::ndt::struct_type::make(field_names, field_types);
}

// Forward declare this now. Include type_functions.hpp at the end.
inline dynd::ndt::type
dynd_make_fixed_dim_type(PyObject *shape, const dynd::ndt::type &element_tp);

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
inline dynd::ndt::type _type_from_numpy_dtype(PyArray_Descr *d,
                                              size_t data_alignment)
{
  dynd::ndt::type dt;

  if (PyDataType_HASSUBARRAY(d)) {
    dt = _type_from_numpy_dtype(d->subarray->base, data_alignment);
    return dynd_make_fixed_dim_type(d->subarray->shape, dt);
  }

  switch (d->type_num) {
  case NPY_BOOL:
    dt = dynd::ndt::make_type<dynd::bool1>();
    break;
  case NPY_BYTE:
    dt = dynd::ndt::make_type<npy_byte>();
    break;
  case NPY_UBYTE:
    dt = dynd::ndt::make_type<npy_ubyte>();
    break;
  case NPY_SHORT:
    dt = dynd::ndt::make_type<npy_short>();
    break;
  case NPY_USHORT:
    dt = dynd::ndt::make_type<npy_ushort>();
    break;
  case NPY_INT:
    dt = dynd::ndt::make_type<npy_int>();
    break;
  case NPY_UINT:
    dt = dynd::ndt::make_type<npy_uint>();
    break;
  case NPY_LONG:
    dt = dynd::ndt::make_type<npy_long>();
    break;
  case NPY_ULONG:
    dt = dynd::ndt::make_type<npy_ulong>();
    break;
  case NPY_LONGLONG:
    dt = dynd::ndt::make_type<npy_longlong>();
    break;
  case NPY_ULONGLONG:
    dt = dynd::ndt::make_type<npy_ulonglong>();
    break;
  case NPY_FLOAT:
    dt = dynd::ndt::make_type<float>();
    break;
  case NPY_DOUBLE:
    dt = dynd::ndt::make_type<double>();
    break;
  case NPY_CFLOAT:
    dt = dynd::ndt::make_type<dynd::complex<float>>();
    break;
  case NPY_CDOUBLE:
    dt = dynd::ndt::make_type<dynd::complex<double>>();
    break;
  case NPY_STRING:
    dt = dynd::ndt::fixed_string_type::make(d->elsize,
                                            dynd::string_encoding_ascii);
    break;
  case NPY_UNICODE:
    dt = dynd::ndt::fixed_string_type::make(d->elsize / 4,
                                            dynd::string_encoding_utf_32);
    break;
  case NPY_VOID:
    dt = make_struct_type_from_numpy_struct(d, data_alignment);
    break;
  case NPY_OBJECT: {
    if (d->fields != NULL && d->fields != Py_None) {
      // Check for an h5py vlen string type (h5py 2.2 style)
      PyObject *vlen_tup = PyMapping_GetItemString(d->fields, "vlen");
      if (vlen_tup != NULL) {
        pyobject_ownref vlen_tup_owner(vlen_tup);
        if (PyTuple_Check(vlen_tup) && PyTuple_GET_SIZE(vlen_tup) == 3) {
          PyObject *type_dict = PyTuple_GET_ITEM(vlen_tup, 2);
          if (PyDict_Check(type_dict)) {
            PyObject *type = PyDict_GetItemString(type_dict, "type");
#if PY_VERSION_HEX < 0x03000000
            if (type == (PyObject *)&PyString_Type ||
                type == (PyObject *)&PyUnicode_Type) {
#else
            if (type == (PyObject *)&PyUnicode_Type) {
#endif
              dt = dynd::ndt::make_type<dynd::ndt::string_type>();
            }
          }
        }
      }
      else {
        PyErr_Clear();
      }
    }
    // Check for an h5py vlen string type (h5py 2.3 style)
    if (d->metadata != NULL && PyDict_Check(d->metadata)) {
      PyObject *type = PyDict_GetItemString(d->metadata, "vlen");
#if PY_VERSION_HEX < 0x03000000
      if (type == (PyObject *)&PyString_Type ||
          type == (PyObject *)&PyUnicode_Type) {
#else
      if (type == (PyObject *)&PyUnicode_Type) {
#endif
        dt = dynd::ndt::make_type<dynd::ndt::string_type>();
      }
    }
    break;
  }
#if NPY_API_VERSION >= 6 // At least NumPy 1.6
  case NPY_DATETIME: {
    // Get the dtype info through the CPython API, slower
    // but lets NumPy's datetime API change without issue.
    pyobject_ownref mod(PyImport_ImportModule("numpy"));
    pyobject_ownref dd(PyObject_CallMethod(mod.get(),
                                           const_cast<char *>("datetime_data"),
                                           const_cast<char *>("O"), d));
    PyObject *unit = PyTuple_GetItem(dd.get(), 0);
    if (unit == NULL) {
      throw std::runtime_error("");
    }
    break;
  }
#endif // At least NumPy 1.6
  default:
    break;
  }

  if (dt.get_id() == dynd::uninitialized_id) {
    std::stringstream ss;
    ss << "unsupported Numpy dtype with type id " << d->type_num;
    throw dynd::type_error(ss.str());
  }

  /*
    if (!PyArray_ISNBO(d->byteorder)) {
      dt = dynd::ndt::new_adapt_type::make(dt);
    }
  */

  // If the data this dtype is for isn't aligned enough,
  // make an unaligned version.
  if (data_alignment != 0 && data_alignment < dt.get_data_alignment()) {
    throw std::runtime_error("unaligned dtype");
  }

  return dt;
}

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
PYDYND_API int _type_from_numpy_scalar_typeobject(PyTypeObject *obj,
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
dynd::nd::array PYDYND_API array_from_numpy_array(PyArrayObject *obj,
                                                  uint32_t access_flags,
                                                  bool always_copy);

dynd::ndt::type PYDYND_API array_from_numpy_array2(PyArrayObject *obj);

// Convenience wrapper for use in Cython where the type has already been
// checked.
inline dynd::nd::array array_from_numpy_array_cast(PyObject *obj,
                                                   uint32_t access_flags,
                                                   bool always_copy)
{
  return array_from_numpy_array(reinterpret_cast<PyArrayObject *>(obj),
                                access_flags, always_copy);
}

/**
 * Creates a dynd::nd::array from a numpy scalar. This always produces
 * a copy.
 *
 * \param obj  The numpy scalar object.
 * \param access_flags  The requested access flags (0 for default).
 */
dynd::nd::array array_from_numpy_scalar(PyObject *obj, uint32_t access_flags);

dynd::ndt::type array_from_numpy_scalar2(PyObject *obj);

/**
 * Returns the numpy kind ('i', 'f', etc) of the array.
 */
inline char numpy_kindchar_of(const dynd::ndt::type &d)
{
  switch (d.get_base_id()) {
  case dynd::bool_kind_id:
    return 'b';
  case dynd::int_kind_id:
    return 'i';
  case dynd::uint_kind_id:
    return 'u';
  case dynd::float_kind_id:
    return 'f';
  case dynd::complex_kind_id:
    return 'c';
  case dynd::string_kind_id:
    if (d.get_id() == dynd::fixed_string_id) {
      const dynd::ndt::base_string_type *esd =
          d.extended<dynd::ndt::base_string_type>();
      switch (esd->get_encoding()) {
      case dynd::string_encoding_ascii:
        return 'S';
      case dynd::string_encoding_utf_32:
        return 'U';
      default:
        break;
      }
    }
    break;
  default:
    break;
  }

  std::stringstream ss;
  ss << "dynd type \"" << d << "\" does not have an equivalent numpy kind";
  throw dynd::type_error(ss.str());
}

// Temporary function introduced to avoid dependency on non-standard
// numpy importing in client modules. Wraps PyArray_DescrCheck.
PYDYND_API bool is_numpy_dtype(PyObject *o);

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

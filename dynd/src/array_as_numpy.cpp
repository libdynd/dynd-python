//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "numpy_interop.hpp"

#if DYND_NUMPY_INTEROP

#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>

#include "array_as_numpy.hpp"
#include "numpy_interop.hpp"
#include "array_functions.hpp"
#include "utility_functions.hpp"
#include "copy_to_numpy_arrfunc.hpp"

#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/base_struct_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/datetime_type.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/property_type.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

using namespace std;
using namespace dynd;
using namespace pydynd;

static int dynd_to_numpy_type_id[builtin_type_id_count] = {
    NPY_NOTYPE,    NPY_BOOL,       NPY_INT8,    NPY_INT16,
    NPY_INT32,     NPY_INT64,      NPY_NOTYPE, // INT128
    NPY_UINT8,     NPY_UINT16,     NPY_UINT32,  NPY_UINT64,
    NPY_NOTYPE,                                             // UINT128
    NPY_FLOAT16,   NPY_FLOAT32,    NPY_FLOAT64, NPY_NOTYPE, // FLOAT128
    NPY_COMPLEX64, NPY_COMPLEX128, NPY_NOTYPE};

static void make_numpy_dtype_for_copy(pyobject_ownref *out_numpy_dtype,
                                      intptr_t ndim, const ndt::type &dt,
                                      const char *arrmeta)
{
  // DyND builtin types
  if (dt.is_builtin()) {
    out_numpy_dtype->reset((PyObject *)PyArray_DescrFromType(
        dynd_to_numpy_type_id[dt.get_type_id()]));
    return;
  }

  switch (dt.get_type_id()) {
  case fixed_string_type_id: {
    const ndt::fixed_string_type *fsd = dt.extended<ndt::fixed_string_type>();
    PyArray_Descr *result;
    switch (fsd->get_encoding()) {
    case string_encoding_ascii:
      result = PyArray_DescrNewFromType(NPY_STRING);
      result->elsize = (int)fsd->get_data_size();
      out_numpy_dtype->reset((PyObject *)result);
      return;
    case string_encoding_utf_32:
      result = PyArray_DescrNewFromType(NPY_UNICODE);
      result->elsize = (int)fsd->get_data_size();
      out_numpy_dtype->reset((PyObject *)result);
      return;
    default:
      // If it's not one of the encodings NumPy supports,
      // use Unicode
      result = PyArray_DescrNewFromType(NPY_UNICODE);
      result->elsize = (int)fsd->get_data_size() * 4 /
                       string_encoding_char_size_table[fsd->get_encoding()];
      out_numpy_dtype->reset((PyObject *)result);
      return;
    }
    break;
  }
  case string_type_id: {
    // Convert variable-length strings into NumPy object arrays
    PyArray_Descr *dtype = PyArray_DescrNewFromType(NPY_OBJECT);
    // Add metadata to the string type being created so that
    // it can round-trip. This metadata is compatible with h5py.
    out_numpy_dtype->reset((PyObject *)dtype);
    if (dtype->metadata == NULL) {
      dtype->metadata = PyDict_New();
    }
    PyDict_SetItemString(dtype->metadata, "vlen", (PyObject *)&PyUnicode_Type);
    return;
  }
  case date_type_id: {
#if NPY_API_VERSION >= 6 // At least NumPy 1.6
    PyArray_Descr *datedt = NULL;
#if PY_VERSION_HEX >= 0x03000000
    pyobject_ownref M8str(PyUnicode_FromString("M8[D]"));
#else
    pyobject_ownref M8str(PyString_FromString("M8[D]"));
#endif
    if (!PyArray_DescrConverter(M8str.get(), &datedt)) {
      throw dynd::type_error("Failed to create NumPy datetime64[D] dtype");
    }
    out_numpy_dtype->reset((PyObject *)datedt);
    return;
#else
    throw runtime_error("NumPy >= 1.6 is required for dynd date type interop");
#endif
  }
  case datetime_type_id: {
#if NPY_API_VERSION >= 6 // At least NumPy 1.6
    PyArray_Descr *datetimedt = NULL;
#if PY_VERSION_HEX >= 0x03000000
    pyobject_ownref M8str(PyUnicode_FromString("M8[us]"));
#else
    pyobject_ownref M8str(PyString_FromString("M8[us]"));
#endif
    if (!PyArray_DescrConverter2(M8str.get(), &datetimedt)) {
      throw dynd::type_error("Failed to create NumPy datetime64[us] dtype");
    }
    out_numpy_dtype->reset((PyObject *)datetimedt);
    return;
#else
    throw runtime_error(
        "NumPy >= 1.6 is required for dynd datetime type interop");
#endif
  }
  case fixed_dim_type_id: {
    if (ndim > 0) {
      const ndt::base_dim_type *bdt = dt.extended<ndt::base_dim_type>();
      make_numpy_dtype_for_copy(out_numpy_dtype, ndim - 1,
                                bdt->get_element_type(),
                                arrmeta + sizeof(fixed_dim_type_arrmeta));
      return;
    } else {
      // If this isn't one of the array dimensions, it maps into
      // a numpy dtype with a shape
      // Build up the shape of the array for NumPy
      pyobject_ownref shape(PyList_New(0));
      ndt::type element_tp = dt;
      while (ndim > 0) {
        const fixed_dim_type_arrmeta *am =
            reinterpret_cast<const fixed_dim_type_arrmeta *>(arrmeta);
        intptr_t dim_size = am->dim_size;
        element_tp = dt.extended<ndt::base_dim_type>()->get_element_type();
        arrmeta += sizeof(fixed_dim_type_arrmeta);
        --ndim;
        if (PyList_Append(shape.get(), PyLong_FromSize_t(dim_size)) < 0) {
          throw runtime_error("propagating python error");
        }
      }
      // Get the numpy dtype of the element
      pyobject_ownref child_numpy_dtype;
      make_numpy_dtype_for_copy(&child_numpy_dtype, 0, element_tp, arrmeta);
      // Create the result numpy dtype
      pyobject_ownref tuple_obj(PyTuple_New(2));
      PyTuple_SET_ITEM(tuple_obj.get(), 0, child_numpy_dtype.release());
      PyTuple_SET_ITEM(tuple_obj.get(), 1, shape.release());

      PyArray_Descr *result = NULL;
      if (!PyArray_DescrConverter(tuple_obj, &result)) {
        throw dynd::type_error(
            "failed to convert dynd type into numpy subarray dtype");
      }
      // Put the final numpy dtype reference in the output
      out_numpy_dtype->reset((PyObject *)result);
      return;
    }
    break;
  }
  case struct_type_id: {
    const ndt::base_struct_type *bs = dt.extended<ndt::base_struct_type>();
    size_t field_count = bs->get_field_count();

    pyobject_ownref names_obj(PyList_New(field_count));
    for (size_t i = 0; i < field_count; ++i) {
      const dynd::string &fn = bs->get_field_name_raw(i);
#if PY_VERSION_HEX >= 0x03000000
      pyobject_ownref name_str(
          PyUnicode_FromStringAndSize(fn.begin(), fn.end() - fn.begin()));
#else
      pyobject_ownref name_str(
          PyString_FromStringAndSize(fn.begin(), fn.end() - fn.begin()));
#endif
      PyList_SET_ITEM(names_obj.get(), i, name_str.release());
    }

    pyobject_ownref formats_obj(PyList_New(field_count));
    pyobject_ownref offsets_obj(PyList_New(field_count));
    size_t standard_offset = 0, standard_alignment = 1;
    for (size_t i = 0; i < field_count; ++i) {
      // Get the numpy dtype of the element
      pyobject_ownref field_numpy_dtype;
      make_numpy_dtype_for_copy(&field_numpy_dtype, 0, bs->get_field_type(i),
                                arrmeta);
      size_t field_alignment =
          ((PyArray_Descr *)field_numpy_dtype.get())->alignment;
      size_t field_size = ((PyArray_Descr *)field_numpy_dtype.get())->elsize;
      standard_offset = inc_to_alignment(standard_offset, field_alignment);
      standard_alignment = max(standard_alignment, field_alignment);
      PyList_SET_ITEM(formats_obj.get(), i, field_numpy_dtype.release());
      PyList_SET_ITEM((PyObject *)offsets_obj, i,
                      PyLong_FromSize_t(standard_offset));
      standard_offset += field_size;
    }
    // Get the full element size
    standard_offset = inc_to_alignment(standard_offset, standard_alignment);
    pyobject_ownref itemsize_obj(PyLong_FromSize_t(standard_offset));

    pyobject_ownref dict_obj(PyDict_New());
    PyDict_SetItemString(dict_obj, "names", names_obj);
    PyDict_SetItemString(dict_obj, "formats", formats_obj);
    PyDict_SetItemString(dict_obj, "offsets", offsets_obj);
    PyDict_SetItemString(dict_obj, "itemsize", itemsize_obj);

    PyArray_Descr *result = NULL;
    if (!PyArray_DescrAlignConverter(dict_obj, &result)) {
      stringstream ss;
      ss << "failed to convert dynd type " << dt
         << " into numpy dtype via dict";
      throw dynd::type_error(ss.str());
    }
    out_numpy_dtype->reset((PyObject *)result);
    return;
  }
  default: {
    break;
  }
  }

  if (dt.get_kind() == expr_kind) {
    // Convert the value type for the copy
    make_numpy_dtype_for_copy(out_numpy_dtype, ndim, dt.value_type(), NULL);
    return;
  }

  // Anything which fell through is an error
  stringstream ss;
  ss << "dynd as_numpy could not convert dynd type ";
  ss << dt;
  ss << " to a numpy dtype";
  throw dynd::type_error(ss.str());
}

static void as_numpy_analysis(pyobject_ownref *out_numpy_dtype,
                              bool *out_requires_copy, intptr_t ndim,
                              const ndt::type &dt, const char *arrmeta)
{
  if (dt.is_builtin()) {
    // DyND builtin types
    out_numpy_dtype->reset((PyObject *)PyArray_DescrFromType(
        dynd_to_numpy_type_id[dt.get_type_id()]));
    return;
  } else if (dt.get_type_id() == view_type_id &&
             dt.operand_type().get_type_id() == fixed_bytes_type_id) {
    // View operation for alignment
    as_numpy_analysis(out_numpy_dtype, out_requires_copy, ndim, dt.value_type(),
                      NULL);
    return;
  }

  switch (dt.get_type_id()) {
  case fixed_string_type_id: {
    const ndt::fixed_string_type *fsd = dt.extended<ndt::fixed_string_type>();
    PyArray_Descr *result;
    switch (fsd->get_encoding()) {
    case string_encoding_ascii:
      result = PyArray_DescrNewFromType(NPY_STRING);
      result->elsize = (int)fsd->get_data_size();
      out_numpy_dtype->reset((PyObject *)result);
      return;
    case string_encoding_utf_32:
      result = PyArray_DescrNewFromType(NPY_UNICODE);
      result->elsize = (int)fsd->get_data_size();
      out_numpy_dtype->reset((PyObject *)result);
      return;
    default:
      out_numpy_dtype->clear();
      *out_requires_copy = true;
      return;
    }
    break;
  }
  case string_type_id: {
    // Convert to numpy object type, requires copy
    out_numpy_dtype->clear();
    *out_requires_copy = true;
    return;
  }
  case date_type_id: {
#if NPY_API_VERSION >= 6 // At least NumPy 1.6
    out_numpy_dtype->clear();
    *out_requires_copy = true;
    return;
#else
    throw runtime_error("NumPy >= 1.6 is required for dynd date type interop");
#endif
  }
  case datetime_type_id: {
#if NPY_API_VERSION >= 6 // At least NumPy 1.6
    out_numpy_dtype->clear();
    *out_requires_copy = true;
    return;
#else
    throw runtime_error("NumPy >= 1.6 is required for dynd date type interop");
#endif
  }
  case property_type_id: {
    const ndt::property_type *pd = dt.extended<ndt::property_type>();
    // Special-case of 'int64 as date' property type, which is binary
    // compatible with NumPy's "M8[D]"
    if (pd->is_reversed_property() &&
        pd->get_value_type().get_type_id() == date_type_id &&
        pd->get_operand_type().get_type_id() == int64_type_id) {
      PyArray_Descr *datedt = NULL;
#if PY_VERSION_HEX >= 0x03000000
      pyobject_ownref M8str(PyUnicode_FromString("M8[D]"));
#else
      pyobject_ownref M8str(PyString_FromString("M8[D]"));
#endif
      if (!PyArray_DescrConverter(M8str.get(), &datedt)) {
        throw dynd::type_error("Failed to create NumPy datetime64[D] dtype");
      }
      out_numpy_dtype->reset((PyObject *)datedt);
      return;
    }
    break;
  }
  case byteswap_type_id: {
    const ndt::base_expr_type *bed = dt.extended<ndt::base_expr_type>();
    // Analyze the unswapped version
    as_numpy_analysis(out_numpy_dtype, out_requires_copy, ndim,
                      bed->get_value_type(), arrmeta);
    pyobject_ownref swapdt(out_numpy_dtype->release());
    // Byteswap the numpy dtype
    out_numpy_dtype->reset((PyObject *)PyArray_DescrNewByteorder(
        (PyArray_Descr *)swapdt.get(), NPY_SWAP));
    return;
  }
  case fixed_dim_type_id: {
    const ndt::base_dim_type *bdt = dt.extended<ndt::base_dim_type>();
    if (ndim > 0) {
      // If this is one of the array dimensions, it simply
      // becomes one of the numpy _array dimensions
      as_numpy_analysis(out_numpy_dtype, out_requires_copy, ndim - 1,
                        bdt->get_element_type(),
                        arrmeta + sizeof(fixed_dim_type_arrmeta));
      return;
    } else {
      // If this isn't one of the array dimensions, it maps into
      // a numpy dtype with a shape
      out_numpy_dtype->clear();
      *out_requires_copy = true;
      return;
    }
    break;
  }
  /*
    case cfixed_dim_type_id: {
      const cfixed_dim_type *fad = dt.extended<cfixed_dim_type>();
      if (ndim > 0) {
        // If this is one of the array dimensions, it simply
        // becomes one of the numpy _array dimensions
        as_numpy_analysis(out_numpy_dtype, out_requires_copy, ndim - 1,
                          fad->get_element_type(),
                          arrmeta + sizeof(cfixed_dim_type_arrmeta));
        return;
      } else {
        // If this isn't one of the array dimensions, it maps into
        // a numpy dtype with a shape
        // Build up the shape of the array for NumPy
        pyobject_ownref shape(PyList_New(0));
        ndt::type element_tp = dt;
        while (ndim > 0) {
          size_t dim_size = 0;
          if (dt.get_type_id() == cfixed_dim_type_id) {
            const cfixed_dim_type *cfd = element_tp.extended<cfixed_dim_type>();
            element_tp = cfd->get_element_type();
            if (cfd->get_data_size() != element_tp.get_data_size() * dim_size) {
              // If it's not C-order, a copy is required
              out_numpy_dtype->clear();
              *out_requires_copy = true;
              return;
            }
          } else {
            stringstream ss;
            ss << "dynd as_numpy could not convert dynd type ";
            ss << dt;
            ss << " to a numpy dtype";
            throw dynd::type_error(ss.str());
          }
          --ndim;
          if (PyList_Append(shape.get(), PyLong_FromSize_t(dim_size)) < 0) {
            throw runtime_error("propagating python error");
          }
        }
        // Get the numpy dtype of the element
        pyobject_ownref child_numpy_dtype;
        as_numpy_analysis(&child_numpy_dtype, out_requires_copy, 0, element_tp,
                          arrmeta);
        if (*out_requires_copy) {
          // If the child required a copy, stop right away
          out_numpy_dtype->clear();
          return;
        }
        // Create the result numpy dtype
        pyobject_ownref tuple_obj(PyTuple_New(2));
        PyTuple_SET_ITEM(tuple_obj.get(), 0, child_numpy_dtype.release());
        PyTuple_SET_ITEM(tuple_obj.get(), 1, shape.release());

        PyArray_Descr *result = NULL;
        if (!PyArray_DescrConverter(tuple_obj, &result)) {
          throw dynd::type_error(
              "failed to convert dynd type into numpy subarray dtype");
        }
        // Put the final numpy dtype reference in the output
        out_numpy_dtype->reset((PyObject *)result);
        return;
      }
      break;
    }
  */
  case struct_type_id: {
    if (dt.get_type_id() == struct_type_id && arrmeta == NULL) {
      // If it's a struct type with no arrmeta, a copy is required
      out_numpy_dtype->clear();
      *out_requires_copy = true;
      return;
    }
    const ndt::base_struct_type *bs = dt.extended<ndt::base_struct_type>();
    const uintptr_t *offsets = bs->get_data_offsets(arrmeta);
    size_t field_count = bs->get_field_count();

    pyobject_ownref names_obj(PyList_New(field_count));
    for (size_t i = 0; i < field_count; ++i) {
      const dynd::string &fn = bs->get_field_name_raw(i);
#if PY_VERSION_HEX >= 0x03000000
      pyobject_ownref name_str(
          PyUnicode_FromStringAndSize(fn.begin(), fn.end() - fn.begin()));
#else
      pyobject_ownref name_str(
          PyString_FromStringAndSize(fn.begin(), fn.end() - fn.begin()));
#endif
      PyList_SET_ITEM(names_obj.get(), i, name_str.release());
    }

    pyobject_ownref formats_obj(PyList_New(field_count));
    for (size_t i = 0; i < field_count; ++i) {
      // Get the numpy dtype of the element
      pyobject_ownref field_numpy_dtype;
      as_numpy_analysis(&field_numpy_dtype, out_requires_copy, 0,
                        bs->get_field_type(i), arrmeta);
      if (*out_requires_copy) {
        // If the field required a copy, stop right away
        out_numpy_dtype->clear();
        return;
      }
      PyList_SET_ITEM(formats_obj.get(), i, field_numpy_dtype.release());
    }

    pyobject_ownref offsets_obj(PyList_New(field_count));
    for (size_t i = 0; i < field_count; ++i) {
      PyList_SET_ITEM((PyObject *)offsets_obj, i,
                      PyLong_FromSize_t(offsets[i]));
    }

    pyobject_ownref dict_obj(PyDict_New());
    PyDict_SetItemString(dict_obj, "names", names_obj);
    PyDict_SetItemString(dict_obj, "formats", formats_obj);
    PyDict_SetItemString(dict_obj, "offsets", offsets_obj);
    if (dt.get_data_size() > 0) {
      pyobject_ownref itemsize_obj(PyLong_FromSize_t(dt.get_data_size()));
      PyDict_SetItemString(dict_obj, "itemsize", itemsize_obj);
    }

    PyArray_Descr *result = NULL;
    if (!PyArray_DescrConverter(dict_obj, &result)) {
      stringstream ss;
      ss << "failed to convert dynd type " << dt
         << " into numpy dtype via dict";
      throw dynd::type_error(ss.str());
    }
    out_numpy_dtype->reset((PyObject *)result);
    return;
  }
  default: {
    break;
  }
  }

  if (dt.get_kind() == expr_kind) {
    // If none of the prior checks caught this expression,
    // a copy is required.
    out_numpy_dtype->clear();
    *out_requires_copy = true;
    return;
  }

  // Anything which fell through is an error
  stringstream ss;
  ss << "dynd as_numpy could not convert dynd type ";
  ss << dt;
  ss << " to a numpy dtype";
  throw dynd::type_error(ss.str());
}

PyObject *pydynd::array_as_numpy(PyObject *a_obj, bool allow_copy)
{
  if (!DyND_PyArray_Check(a_obj)) {
    throw runtime_error("can only call dynd's as_numpy on dynd arrays");
  }
  nd::array a = ((DyND_PyArrayObject *)a_obj)->v;
  if (a.get() == NULL) {
    throw runtime_error("cannot convert NULL dynd array to numpy");
  }

  // If a copy is allowed, convert the builtin scalars to NumPy scalars
  if (allow_copy && a.get_type().is_scalar()) {
    pyobject_ownref result;
    switch (a.get_type().get_type_id()) {
    case uninitialized_type_id:
      throw runtime_error("cannot convert uninitialized dynd array to numpy");
    case void_type_id:
      throw runtime_error("cannot convert void dynd array to numpy");
    case bool_type_id:
      if (*a.cdata()) {
        Py_INCREF(PyArrayScalar_True);
        result.reset(PyArrayScalar_True);
      } else {
        Py_INCREF(PyArrayScalar_False);
        result.reset(PyArrayScalar_False);
      }
      break;
    case int8_type_id:
      result.reset(PyArrayScalar_New(Int8));
      PyArrayScalar_ASSIGN(result.get(), Int8,
                           *reinterpret_cast<const int8_t *>(a.cdata()));
      break;
    case int16_type_id:
      result.reset(PyArrayScalar_New(Int16));
      PyArrayScalar_ASSIGN(result.get(), Int16,
                           *reinterpret_cast<const int16_t *>(a.cdata()));
      break;
    case int32_type_id:
      result.reset(PyArrayScalar_New(Int32));
      PyArrayScalar_ASSIGN(result.get(), Int32,
                           *reinterpret_cast<const int32_t *>(a.cdata()));
      break;
    case int64_type_id:
      result.reset(PyArrayScalar_New(Int64));
      PyArrayScalar_ASSIGN(result.get(), Int64,
                           *reinterpret_cast<const int64_t *>(a.cdata()));
      break;
    case uint8_type_id:
      result.reset(PyArrayScalar_New(UInt8));
      PyArrayScalar_ASSIGN(result.get(), UInt8,
                           *reinterpret_cast<const uint8_t *>(a.cdata()));
      break;
    case uint16_type_id:
      result.reset(PyArrayScalar_New(UInt16));
      PyArrayScalar_ASSIGN(result.get(), UInt16,
                           *reinterpret_cast<const uint16_t *>(a.cdata()));
      break;
    case uint32_type_id:
      result.reset(PyArrayScalar_New(UInt32));
      PyArrayScalar_ASSIGN(result.get(), UInt32,
                           *reinterpret_cast<const uint32_t *>(a.cdata()));
      break;
    case uint64_type_id:
      result.reset(PyArrayScalar_New(UInt64));
      PyArrayScalar_ASSIGN(result.get(), UInt64,
                           *reinterpret_cast<const uint64_t *>(a.cdata()));
      break;
    case float32_type_id:
      result.reset(PyArrayScalar_New(Float32));
      PyArrayScalar_ASSIGN(result.get(), Float32,
                           *reinterpret_cast<const float *>(a.cdata()));
      break;
    case float64_type_id:
      result.reset(PyArrayScalar_New(Float64));
      PyArrayScalar_ASSIGN(result.get(), Float64,
                           *reinterpret_cast<const double *>(a.cdata()));
      break;
    case complex_float32_type_id:
      result.reset(PyArrayScalar_New(Complex64));
      PyArrayScalar_VAL(result.get(), Complex64).real =
          reinterpret_cast<const float *>(a.cdata())[0];
      PyArrayScalar_VAL(result.get(), Complex64).imag =
          reinterpret_cast<const float *>(a.cdata())[1];
      break;
    case complex_float64_type_id:
      result.reset(PyArrayScalar_New(Complex128));
      PyArrayScalar_VAL(result.get(), Complex128).real =
          reinterpret_cast<const double *>(a.cdata())[0];
      PyArrayScalar_VAL(result.get(), Complex128).imag =
          reinterpret_cast<const double *>(a.cdata())[1];
      break;
    case date_type_id: {
#if NPY_API_VERSION >= 6 // At least NumPy 1.6
      int32_t dateval = *reinterpret_cast<const int32_t *>(a.cdata());
      result.reset(PyArrayScalar_New(Datetime));

      PyArrayScalar_VAL(result.get(), Datetime) =
          (dateval == DYND_DATE_NA) ? NPY_DATETIME_NAT : dateval;
      PyArray_DatetimeMetaData &meta =
          ((PyDatetimeScalarObject *)result.get())->obmeta;
      meta.base = NPY_FR_D;
      meta.num = 1;
#if NPY_API_VERSION == 6 // Only for NumPy 1.6
      meta.den = 1;
      meta.events = 1;
#endif
      break;
#else
      throw runtime_error(
          "NumPy >= 1.6 is required for dynd date type interop");
#endif
    }
    case datetime_type_id: {
#if NPY_API_VERSION >= 6 // At least NumPy 1.6
      int64_t datetimeval = *reinterpret_cast<const int64_t *>(a.cdata());
      if (datetimeval < 0) {
        datetimeval -= DYND_TICKS_PER_MICROSECOND - 1;
      }
      datetimeval /= DYND_TICKS_PER_MICROSECOND;
      result.reset(PyArrayScalar_New(Datetime));

      PyArrayScalar_VAL(result.get(), Datetime) =
          (datetimeval == DYND_DATETIME_NA) ? NPY_DATETIME_NAT : datetimeval;
      PyArray_DatetimeMetaData &meta =
          ((PyDatetimeScalarObject *)result.get())->obmeta;
      meta.base = NPY_FR_us;
      meta.num = 1;
#if NPY_API_VERSION == 6 // Only for NumPy 1.6
      meta.den = 1;
      meta.events = 1;
#endif
      break;
#else
      throw runtime_error(
          "NumPy >= 1.6 is required for dynd datetime type interop");
#endif
    }
    default: {
      // Because 'allow_copy' is true
      // we can evaluate any expressions and
      // make copies of strings
      if (a.get_type().get_kind() == expr_kind) {
        // If it's an expression kind
        pyobject_ownref n_tmp(DyND_PyWrapper_New(a.eval()));
        return array_as_numpy(n_tmp.get(), true);
      } else if (a.get_type().get_kind() == string_kind) {
        // If it's a string kind, return it as a Python unicode
        return array_as_py(a, false);
      }
      stringstream ss;
      ss << "dynd as_numpy could not convert dynd type ";
      ss << a.get_type();
      ss << " to a numpy dtype";
      throw dynd::type_error(ss.str());
    }
    }
    return result.release();
  }

  if (a.get_type().get_type_id() == var_dim_type_id) {
    // If it's a var_dim, view it as fixed then try again
    pyobject_ownref n_tmp(DyND_PyWrapper_New(a.view(ndt::make_fixed_dim(
        a.get_dim_size(),
        a.get_type().extended<ndt::base_dim_type>()->get_element_type()))));
    return array_as_numpy(n_tmp.get(), allow_copy);
  }
  // TODO: Handle pointer type nicely as well
  // n.get_type().get_type_id() == pointer_type_id

  // Do a recursive analysis of the dynd array for how to
  // convert it to NumPy
  bool requires_copy = false;
  pyobject_ownref numpy_dtype;
  size_t ndim = a.get_ndim();
  dimvector shape(ndim), strides(ndim);

  a.get_shape(shape.get());
  a.get_strides(strides.get());
  as_numpy_analysis(&numpy_dtype, &requires_copy, ndim, a.get_type(),
                    a.get()->metadata());
  if (requires_copy) {
    if (!allow_copy) {
      stringstream ss;
      ss << "cannot view dynd array with dtype " << a.get_type();
      ss << " as numpy without making a copy";
      throw dynd::type_error(ss.str());
    }
    make_numpy_dtype_for_copy(&numpy_dtype, ndim, a.get_type(),
                              a.get()->metadata());

    // Rebuild the strides so that the copy follows 'KEEPORDER'
    intptr_t element_size = ((PyArray_Descr *)numpy_dtype.get())->elsize;
    if (ndim == 1) {
      strides[0] = element_size;
    } else if (ndim > 1) {
      shortvector<int> axis_perm(ndim);
      strides_to_axis_perm(ndim, strides.get(), axis_perm.get());
      axis_perm_to_strides(ndim, axis_perm.get(), shape.get(), element_size,
                           strides.get());
    }

    // Create a new NumPy array, and copy from the dynd array
    pyobject_ownref result(PyArray_NewFromDescr(
        &PyArray_Type, (PyArray_Descr *)numpy_dtype.release(), (int)ndim,
        shape.get(), strides.get(), NULL, 0, NULL));
    array_copy_to_numpy((PyArrayObject *)result.get(), a.get_type(),
                        a.get()->metadata(), a.cdata(),
                        &eval::default_eval_context);

    // Return the NumPy array
    return result.release();
  } else {
    // Create a view directly to the dynd array
    pyobject_ownref result(PyArray_NewFromDescr(
        &PyArray_Type, (PyArray_Descr *)numpy_dtype.release(), (int)ndim,
        shape.get(), strides.get(), a.get()->data,
        ((a.get_flags() & nd::write_access_flag) ? NPY_ARRAY_WRITEABLE : 0) |
            NPY_ARRAY_ALIGNED,
        NULL));

#if NPY_API_VERSION >= 7 // At least NumPy 1.7
    Py_INCREF(a_obj);
    if (PyArray_SetBaseObject((PyArrayObject *)result.get(), a_obj) < 0) {
      throw runtime_error("propagating python exception");
    }
#else
    PyArray_BASE(result.get()) = n_obj;
    Py_INCREF(n_obj);
#endif
    return result.release();
  }
}

#endif // NUMPY_INTEROP

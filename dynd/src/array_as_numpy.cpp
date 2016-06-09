//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "numpy_interop.hpp"

#if DYND_NUMPY_INTEROP

#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>

#include "array_as_numpy.hpp"
#include "array_functions.hpp"
#include "assign.hpp"
#include "numpy_interop.hpp"
#include "types/pyobject_type.hpp"
#include "utility_functions.hpp"

#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/struct_type.hpp>

using namespace std;
using namespace dynd;
using namespace pydynd;

static int dynd_to_numpy_id(dynd::type_id_t id)
{
  switch (id) {
  case uninitialized_id:
    return NPY_NOTYPE;
  case bool_id:
    return NPY_BOOL;
  case int8_id:
    return NPY_INT8;
  case int16_id:
    return NPY_INT16;
  case int32_id:
    return NPY_INT32;
  case int64_id:
    return NPY_INT64;
  case int128_id:
    return NPY_NOTYPE;
  case uint8_id:
    return NPY_UINT8;
  case uint16_id:
    return NPY_UINT16;
  case uint32_id:
    return NPY_UINT32;
  case uint64_id:
    return NPY_UINT64;
  case uint128_id:
    return NPY_NOTYPE;
  case float16_id:
    return NPY_FLOAT16;
  case float32_id:
    return NPY_FLOAT32;
  case float64_id:
    return NPY_FLOAT64;
  case float128_id:
    return NPY_NOTYPE;
  case complex_float32_id:
    return NPY_COMPLEX64;
  case complex_float64_id:
    return NPY_COMPLEX128;
  default:
    return NPY_NOTYPE;
  }
}

static void make_numpy_dtype_for_copy(py_ref *out_numpy_dtype, intptr_t ndim, const ndt::type &dt, const char *arrmeta)
{
  // DyND builtin types
  if (dt.is_builtin()) {
    *out_numpy_dtype = capture_if_not_null((PyObject *)PyArray_DescrFromType(dynd_to_numpy_id(dt.get_id())));
    return;
  }

  switch (dt.get_id()) {
  case fixed_string_id: {
    const ndt::fixed_string_type *fsd = dt.extended<ndt::fixed_string_type>();
    PyArray_Descr *result;
    switch (fsd->get_encoding()) {
    case string_encoding_ascii:
      result = PyArray_DescrNewFromType(NPY_STRING);
      result->elsize = (int)fsd->get_data_size();
      *out_numpy_dtype = capture_if_not_null((PyObject *)result);
      return;
    case string_encoding_utf_32:
      result = PyArray_DescrNewFromType(NPY_UNICODE);
      result->elsize = (int)fsd->get_data_size();
      *out_numpy_dtype = capture_if_not_null((PyObject *)result);
      return;
    default:
      // If it's not one of the encodings NumPy supports,
      // use Unicode
      result = PyArray_DescrNewFromType(NPY_UNICODE);
      result->elsize = (int)fsd->get_data_size() * 4 / string_encoding_char_size_table[fsd->get_encoding()];
      *out_numpy_dtype = capture_if_not_null((PyObject *)result);
      return;
    }
    break;
  }
  case string_id: {
    // Convert variable-length strings into NumPy object arrays
    PyArray_Descr *dtype = PyArray_DescrNewFromType(NPY_OBJECT);
    // Add metadata to the string type being created so that
    // it can round-trip. This metadata is compatible with h5py.
    *out_numpy_dtype = capture_if_not_null((PyObject *)dtype);
    if (dtype->metadata == NULL) {
      dtype->metadata = PyDict_New();
    }
    PyDict_SetItemString(dtype->metadata, "vlen", (PyObject *)&PyUnicode_Type);
    return;
  }
  case fixed_dim_id: {
    if (ndim > 0) {
      const ndt::base_dim_type *bdt = dt.extended<ndt::base_dim_type>();
      make_numpy_dtype_for_copy(out_numpy_dtype, ndim - 1, bdt->get_element_type(),
                                arrmeta + sizeof(fixed_dim_type_arrmeta));
      return;
    }
    else {
      // If this isn't one of the array dimensions, it maps into
      // a numpy dtype with a shape
      // Build up the shape of the array for NumPy
      py_ref shape = capture_if_not_null(PyList_New(0));
      ndt::type element_tp = dt;
      while (ndim > 0) {
        const fixed_dim_type_arrmeta *am = reinterpret_cast<const fixed_dim_type_arrmeta *>(arrmeta);
        intptr_t dim_size = am->dim_size;
        element_tp = dt.extended<ndt::base_dim_type>()->get_element_type();
        arrmeta += sizeof(fixed_dim_type_arrmeta);
        --ndim;
        if (PyList_Append(shape.get(), PyLong_FromSize_t(dim_size)) < 0) {
          throw runtime_error("propagating python error");
        }
      }
      // Get the numpy dtype of the element
      py_ref child_numpy_dtype;
      make_numpy_dtype_for_copy(&child_numpy_dtype, 0, element_tp, arrmeta);
      // Create the result numpy dtype
      py_ref tuple_obj = capture_if_not_null(PyTuple_New(2));
      PyTuple_SET_ITEM(tuple_obj.get(), 0, release(std::move(child_numpy_dtype)));
      PyTuple_SET_ITEM(tuple_obj.get(), 1, release(std::move(shape)));

      PyArray_Descr *result = NULL;
      if (!PyArray_DescrConverter(tuple_obj.get(), &result)) {
        throw dynd::type_error("failed to convert dynd type into numpy subarray dtype");
      }
      // Put the final numpy dtype reference in the output
      *out_numpy_dtype = capture_if_not_null((PyObject *)result);
      return;
    }
    break;
  }
  case struct_id: {
    const ndt::struct_type *bs = dt.extended<ndt::struct_type>();
    size_t field_count = bs->get_field_count();

    py_ref names_obj = capture_if_not_null(PyList_New(field_count));
    for (size_t i = 0; i < field_count; ++i) {
      const dynd::string &fn = bs->get_field_name(i);
#if PY_VERSION_HEX >= 0x03000000
      py_ref name_str = capture_if_not_null(PyUnicode_FromStringAndSize(fn.begin(), fn.end() - fn.begin()));
#else
      py_ref name_str = capture_if_not_null(PyString_FromStringAndSize(fn.begin(), fn.end() - fn.begin()));
#endif
      PyList_SET_ITEM(names_obj.get(), i, release(std::move(name_str)));
    }

    py_ref formats_obj = capture_if_not_null(PyList_New(field_count));
    py_ref offsets_obj = capture_if_not_null(PyList_New(field_count));
    size_t standard_offset = 0, standard_alignment = 1;
    for (size_t i = 0; i < field_count; ++i) {
      // Get the numpy dtype of the element
      py_ref field_numpy_dtype;
      make_numpy_dtype_for_copy(&field_numpy_dtype, 0, bs->get_field_type(i), arrmeta);
      size_t field_alignment = ((PyArray_Descr *)field_numpy_dtype.get())->alignment;
      size_t field_size = ((PyArray_Descr *)field_numpy_dtype.get())->elsize;
      standard_offset = inc_to_alignment(standard_offset, field_alignment);
      standard_alignment = max(standard_alignment, field_alignment);
      PyList_SET_ITEM(formats_obj.get(), i, release(std::move(field_numpy_dtype)));
      PyList_SET_ITEM(offsets_obj.get(), i, PyLong_FromSize_t(standard_offset));
      standard_offset += field_size;
    }
    // Get the full element size
    standard_offset = inc_to_alignment(standard_offset, standard_alignment);
    py_ref itemsize_obj = capture_if_not_null(PyLong_FromSize_t(standard_offset));

    py_ref dict_obj = capture_if_not_null(PyDict_New());
    PyDict_SetItemString(dict_obj.get(), "names", names_obj.get());
    PyDict_SetItemString(dict_obj.get(), "formats", formats_obj.get());
    PyDict_SetItemString(dict_obj.get(), "offsets", offsets_obj.get());
    PyDict_SetItemString(dict_obj.get(), "itemsize", itemsize_obj.get());

    PyArray_Descr *result = NULL;
    if (!PyArray_DescrAlignConverter(dict_obj.get(), &result)) {
      stringstream ss;
      ss << "failed to convert dynd type " << dt << " into numpy dtype via dict";
      throw dynd::type_error(ss.str());
    }
    *out_numpy_dtype = capture_if_not_null((PyObject *)result);
    return;
  }
  default: {
    break;
  }
  }

  if (dt.get_base_id() == expr_kind_id) {
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

static void as_numpy_analysis(py_ref *out_numpy_dtype, bool *out_requires_copy, intptr_t ndim, const ndt::type &dt,
                              const char *arrmeta)
{
  if (dt.is_builtin()) {
    // DyND builtin types
    *out_numpy_dtype = capture_if_not_null((PyObject *)PyArray_DescrFromType(dynd_to_numpy_id(dt.get_id())));
    return;
  }
  switch (dt.get_id()) {
  case fixed_string_id: {
    const ndt::fixed_string_type *fsd = dt.extended<ndt::fixed_string_type>();
    PyArray_Descr *result;
    switch (fsd->get_encoding()) {
    case string_encoding_ascii:
      result = PyArray_DescrNewFromType(NPY_STRING);
      result->elsize = (int)fsd->get_data_size();
      *out_numpy_dtype = capture_if_not_null((PyObject *)result);
      return;
    case string_encoding_utf_32:
      result = PyArray_DescrNewFromType(NPY_UNICODE);
      result->elsize = (int)fsd->get_data_size();
      *out_numpy_dtype = capture_if_not_null((PyObject *)result);
      return;
    default:
      *out_numpy_dtype = py_ref(Py_None, false);
      *out_requires_copy = true;
      return;
    }
    break;
  }
  case string_id: {
    // Convert to numpy object type, requires copy
    *out_numpy_dtype = py_ref(Py_None, false);
    *out_requires_copy = true;
    return;
  }
  case fixed_dim_id: {
    const ndt::base_dim_type *bdt = dt.extended<ndt::base_dim_type>();
    if (ndim > 0) {
      // If this is one of the array dimensions, it simply
      // becomes one of the numpy _array dimensions
      as_numpy_analysis(out_numpy_dtype, out_requires_copy, ndim - 1, bdt->get_element_type(),
                        arrmeta + sizeof(fixed_dim_type_arrmeta));
      return;
    }
    else {
      // If this isn't one of the array dimensions, it maps into
      // a numpy dtype with a shape
      *out_numpy_dtype = py_ref(Py_None, false);
      *out_requires_copy = true;
      return;
    }
    break;
  }
  /*
    case cfixed_dim_id: {
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
        py_ref shape = capture_if_not_null(PyList_New(0));
        ndt::type element_tp = dt;
        while (ndim > 0) {
          size_t dim_size = 0;
          if (dt.get_id() == cfixed_dim_id) {
            const cfixed_dim_type *cfd =
    element_tp.extended<cfixed_dim_type>();
            element_tp = cfd->get_element_type();
            if (cfd->get_data_size() != element_tp.get_data_size() * dim_size)
    {
              // If it's not C-order, a copy is required
              *out_numpy_dtype = py_ref(Py_None, false);
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
        py_ref child_numpy_dtype;
        as_numpy_analysis(&child_numpy_dtype, out_requires_copy, 0,
    element_tp,
                          arrmeta);
        if (*out_requires_copy) {
          // If the child required a copy, stop right away
          *out_numpy_dtype = py_ref(Py_None, false);
          return;
        }
        // Create the result numpy dtype
        py_ref tuple_obj = capture_if_not_null(PyTuple_New(2));
        PyTuple_SET_ITEM(tuple_obj.get(), 0, release(std::move(child_numpy_dtype)));
        PyTuple_SET_ITEM(tuple_obj.get(), 1, release(std::move(shape)));

        PyArray_Descr *result = NULL;
        if (!PyArray_DescrConverter(tuple_obj, &result)) {
          throw dynd::type_error(
              "failed to convert dynd type into numpy subarray dtype");
        }
        // Put the final numpy dtype reference in the output
        *out_numpy_dtype = capture_if_not_null((PyObject *)result);
        return;
      }
      break;
    }
  */
  case struct_id: {
    if (dt.get_id() == struct_id && arrmeta == NULL) {
      // If it's a struct type with no arrmeta, a copy is required
      *out_numpy_dtype = py_ref(Py_None, false);
      *out_requires_copy = true;
      return;
    }
    const ndt::struct_type *bs = dt.extended<ndt::struct_type>();
    const uintptr_t *offsets = reinterpret_cast<const uintptr_t *>(arrmeta);
    size_t field_count = bs->get_field_count();

    py_ref names_obj = capture_if_not_null(PyList_New(field_count));
    for (size_t i = 0; i < field_count; ++i) {
      const dynd::string &fn = bs->get_field_name(i);
#if PY_VERSION_HEX >= 0x03000000
      py_ref name_str = capture_if_not_null(PyUnicode_FromStringAndSize(fn.begin(), fn.end() - fn.begin()));
#else
      py_ref name_str = capture_if_not_null(PyString_FromStringAndSize(fn.begin(), fn.end() - fn.begin()));
#endif
      PyList_SET_ITEM(names_obj.get(), i, release(std::move(name_str)));
    }

    py_ref formats_obj = capture_if_not_null(PyList_New(field_count));
    for (size_t i = 0; i < field_count; ++i) {
      // Get the numpy dtype of the element
      py_ref field_numpy_dtype;
      as_numpy_analysis(&field_numpy_dtype, out_requires_copy, 0, bs->get_field_type(i), arrmeta);
      if (*out_requires_copy) {
        // If the field required a copy, stop right away
        *out_numpy_dtype = py_ref(Py_None, false);
        return;
      }
      PyList_SET_ITEM(formats_obj.get(), i, release(std::move(field_numpy_dtype)));
    }

    py_ref offsets_obj = capture_if_not_null(PyList_New(field_count));
    for (size_t i = 0; i < field_count; ++i) {
      PyList_SET_ITEM(offsets_obj.get(), i, PyLong_FromSize_t(offsets[i]));
    }

    py_ref dict_obj = capture_if_not_null(PyDict_New());
    PyDict_SetItemString(dict_obj.get(), "names", names_obj.get());
    PyDict_SetItemString(dict_obj.get(), "formats", formats_obj.get());
    PyDict_SetItemString(dict_obj.get(), "offsets", offsets_obj.get());
    if (dt.get_data_size() > 0) {
      py_ref itemsize_obj = capture_if_not_null(PyLong_FromSize_t(dt.get_data_size()));
      PyDict_SetItemString(dict_obj.get(), "itemsize", itemsize_obj.get());
    }

    PyArray_Descr *result = NULL;
    if (!PyArray_DescrConverter(dict_obj.get(), &result)) {
      stringstream ss;
      ss << "failed to convert dynd type " << dt << " into numpy dtype via dict";
      throw dynd::type_error(ss.str());
    }
    *out_numpy_dtype = capture_if_not_null((PyObject *)result);
    return;
  }
  default: {
    break;
  }
  }

  if (dt.get_base_id() == expr_kind_id) {
    // If none of the prior checks caught this expression,
    // a copy is required.
    *out_numpy_dtype = py_ref(Py_None, false);
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
  if (!PyObject_TypeCheck(a_obj, pydynd::get_array_pytypeobject())) {
    throw runtime_error("can only call dynd's as_numpy on dynd arrays");
  }
  nd::array a = pydynd::array_to_cpp_ref(a_obj);
  if (a.get() == NULL) {
    throw runtime_error("cannot convert NULL dynd array to numpy");
  }

  // If a copy is allowed, convert the builtin scalars to NumPy scalars
  if (allow_copy && a.get_type().is_scalar()) {
    py_ref result;
    switch (a.get_type().get_id()) {
    case uninitialized_id:
      throw runtime_error("cannot convert uninitialized dynd array to numpy");
    case void_id:
      throw runtime_error("cannot convert void dynd array to numpy");
    case bool_id:
      if (*a.cdata()) {
        Py_INCREF(PyArrayScalar_True);
        result = capture_if_not_null(PyArrayScalar_True);
      }
      else {
        Py_INCREF(PyArrayScalar_False);
        result = capture_if_not_null(PyArrayScalar_False);
      }
      break;
    case int8_id:
      result = capture_if_not_null(PyArrayScalar_New(Int8));
      PyArrayScalar_ASSIGN(result.get(), Int8, *reinterpret_cast<const int8_t *>(a.cdata()));
      break;
    case int16_id:
      result = capture_if_not_null(PyArrayScalar_New(Int16));
      PyArrayScalar_ASSIGN(result.get(), Int16, *reinterpret_cast<const int16_t *>(a.cdata()));
      break;
    case int32_id:
      result = capture_if_not_null(PyArrayScalar_New(Int32));
      PyArrayScalar_ASSIGN(result.get(), Int32, *reinterpret_cast<const int32_t *>(a.cdata()));
      break;
    case int64_id:
      result = capture_if_not_null(PyArrayScalar_New(Int64));
      PyArrayScalar_ASSIGN(result.get(), Int64, *reinterpret_cast<const int64_t *>(a.cdata()));
      break;
    case uint8_id:
      result = capture_if_not_null(PyArrayScalar_New(UInt8));
      PyArrayScalar_ASSIGN(result.get(), UInt8, *reinterpret_cast<const uint8_t *>(a.cdata()));
      break;
    case uint16_id:
      result = capture_if_not_null(PyArrayScalar_New(UInt16));
      PyArrayScalar_ASSIGN(result.get(), UInt16, *reinterpret_cast<const uint16_t *>(a.cdata()));
      break;
    case uint32_id:
      result = capture_if_not_null(PyArrayScalar_New(UInt32));
      PyArrayScalar_ASSIGN(result.get(), UInt32, *reinterpret_cast<const uint32_t *>(a.cdata()));
      break;
    case uint64_id:
      result = capture_if_not_null(PyArrayScalar_New(UInt64));
      PyArrayScalar_ASSIGN(result.get(), UInt64, *reinterpret_cast<const uint64_t *>(a.cdata()));
      break;
    case float32_id:
      result = capture_if_not_null(PyArrayScalar_New(Float32));
      PyArrayScalar_ASSIGN(result.get(), Float32, *reinterpret_cast<const float *>(a.cdata()));
      break;
    case float64_id:
      result = capture_if_not_null(PyArrayScalar_New(Float64));
      PyArrayScalar_ASSIGN(result.get(), Float64, *reinterpret_cast<const double *>(a.cdata()));
      break;
    case complex_float32_id:
      result = capture_if_not_null(PyArrayScalar_New(Complex64));
      PyArrayScalar_VAL(result.get(), Complex64).real = reinterpret_cast<const float *>(a.cdata())[0];
      PyArrayScalar_VAL(result.get(), Complex64).imag = reinterpret_cast<const float *>(a.cdata())[1];
      break;
    case complex_float64_id:
      result = capture_if_not_null(PyArrayScalar_New(Complex128));
      PyArrayScalar_VAL(result.get(), Complex128).real = reinterpret_cast<const double *>(a.cdata())[0];
      PyArrayScalar_VAL(result.get(), Complex128).imag = reinterpret_cast<const double *>(a.cdata())[1];
      break;
    default: {
      // Because 'allow_copy' is true
      // we can evaluate any expressions and
      // make copies of strings
      if (a.get_type().get_base_id() == expr_kind_id) {
        // If it's an expression kind
        py_ref n_tmp = capture_if_not_null(pydynd::array_from_cpp(a.eval()));
        return array_as_numpy(n_tmp.get(), true);
      }
      else if (a.get_type().get_base_id() == string_kind_id) {
        nd::array res = nd::empty(ndt::make_type<pyobject_type>());
        res.assign(a);
        PyObject *res_obj = *reinterpret_cast<PyObject **>(res.data());
        Py_INCREF(res_obj);
        return res_obj;
      }
      stringstream ss;
      ss << "dynd as_numpy could not convert dynd type ";
      ss << a.get_type();
      ss << " to a numpy dtype";
      throw dynd::type_error(ss.str());
    }
    }
    return release(std::move(result));
  }

  if (a.get_type().get_id() == var_dim_id) {
    // If it's a var_dim, view it as fixed then try again
    py_ref n_tmp = capture_if_not_null(pydynd::array_from_cpp(a.view(
        ndt::make_fixed_dim(a.get_dim_size(), a.get_type().extended<ndt::base_dim_type>()->get_element_type()))));
    return array_as_numpy(n_tmp.get(), allow_copy);
  }
  // TODO: Handle pointer type nicely as well
  // n.get_type().get_id() == pointer_id

  // Do a recursive analysis of the dynd array for how to
  // convert it to NumPy
  bool requires_copy = false;
  py_ref numpy_dtype;
  size_t ndim = a.get_ndim();
  dimvector shape(ndim), strides(ndim);

  a.get_shape(shape.get());
  a.get_strides(strides.get());
  as_numpy_analysis(&numpy_dtype, &requires_copy, ndim, a.get_type(), a.get()->metadata());
  if (requires_copy) {
    if (!allow_copy) {
      stringstream ss;
      ss << "cannot view dynd array with dtype " << a.get_type();
      ss << " as numpy without making a copy";
      throw dynd::type_error(ss.str());
    }
    make_numpy_dtype_for_copy(&numpy_dtype, ndim, a.get_type(), a.get()->metadata());

    // Rebuild the strides so that the copy follows 'KEEPORDER'
    intptr_t element_size = ((PyArray_Descr *)numpy_dtype.get())->elsize;
    if (ndim == 1) {
      strides[0] = element_size;
    }
    else if (ndim > 1) {
      shortvector<int> axis_perm(ndim);
      strides_to_axis_perm(ndim, strides.get(), axis_perm.get());
      axis_perm_to_strides(ndim, axis_perm.get(), shape.get(), element_size, strides.get());
    }

    // Create a new NumPy array, and copy from the dynd array
    py_ref result = capture_if_not_null(
        PyArray_NewFromDescr(&PyArray_Type, reinterpret_cast<PyArray_Descr *>(release(std::move(numpy_dtype))),
                             (int)ndim, shape.get(), strides.get(), NULL, 0, NULL));
    array_copy_to_numpy((PyArrayObject *)result.get(), a.get_type(), a.get()->metadata(), a.cdata());

    // Return the NumPy array
    return release(std::move(result));
  }
  else {
    // Create a view directly to the dynd array
    py_ref result = capture_if_not_null(PyArray_NewFromDescr(
        &PyArray_Type, reinterpret_cast<PyArray_Descr *>(release(std::move(numpy_dtype))), (int)ndim, shape.get(),
        strides.get(), const_cast<char *>(a.cdata()),
        ((a.get_flags() & nd::write_access_flag) ? NPY_ARRAY_WRITEABLE : 0) | NPY_ARRAY_ALIGNED, NULL));

#if NPY_API_VERSION >= 7 // At least NumPy 1.7
    Py_INCREF(a_obj);
    if (PyArray_SetBaseObject((PyArrayObject *)result.get(), a_obj) < 0) {
      throw runtime_error("propagating python exception");
    }
#else
    PyArray_BASE(result.get()) = n_obj;
    Py_INCREF(n_obj);
#endif
    return release(std::move(result));
  }
}

#endif // NUMPY_INTEROP

//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>
#include <datetime.h>

#include <dynd/callable.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/memblock/external_memory_block.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/option.hpp>
#include <dynd/type_promotion.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/substitute_shape.hpp>
#include <dynd/types/type_type.hpp>
#include <dynd/types/var_dim_type.hpp>

#include "array_conversions.hpp"
#include "array_from_py.hpp"
#include "array_functions.hpp"
#include "numpy_interop.hpp"
#include "type_deduction.hpp"
#include "type_functions.hpp"
#include "types/pyobject_type.hpp"
#include "utility_functions.hpp"

using namespace std;
using namespace dynd;
using namespace pydynd;

void pydynd::init_array_from_py()
{
  // Initialize the pydatetime API
  PyDateTime_IMPORT;
}

typedef void (*convert_one_pyscalar_function_t)(const ndt::type &tp, const char *arrmeta, char *out, PyObject *obj);

inline void convert_one_pyscalar_bool(const ndt::type &tp, const char *arrmeta, char *out, PyObject *obj)
{
  *out = (PyObject_IsTrue(obj) != 0);
}

inline void convert_one_pyscalar_int32(const ndt::type &tp, const char *arrmeta, char *out, PyObject *obj)
{
#if PY_VERSION_HEX >= 0x03000000
  int32_t value = static_cast<int32_t>(PyLong_AsLong(obj));
#else
  int32_t value = static_cast<int32_t>(PyInt_AsLong(obj));
#endif
  if (value == -1 && PyErr_Occurred()) {
    throw std::exception();
  }
  *reinterpret_cast<int32_t *>(out) = value;
}

inline void convert_one_pyscalar_int64(const ndt::type &tp, const char *arrmeta, char *out, PyObject *obj)
{
  int64_t value = PyLong_AsLongLong(obj);
  if (value == -1 && PyErr_Occurred()) {
    throw std::exception();
  }
  *reinterpret_cast<int64_t *>(out) = value;
}

inline void convert_one_pyscalar_float32(const ndt::type &tp, const char *arrmeta, char *out, PyObject *obj)
{
  double value = PyFloat_AsDouble(obj);
  if (value == -1 && PyErr_Occurred()) {
    throw std::exception();
  }
  *reinterpret_cast<float *>(out) = static_cast<float>(value);
}

inline void convert_one_pyscalar_float64(const ndt::type &tp, const char *arrmeta, char *out, PyObject *obj)
{
  double value = PyFloat_AsDouble(obj);
  if (value == -1 && PyErr_Occurred()) {
    throw std::exception();
  }
  *reinterpret_cast<double *>(out) = value;
}

inline void convert_one_pyscalar_cdouble(const ndt::type &tp, const char *arrmeta, char *out, PyObject *obj)
{
  double value_real = PyComplex_RealAsDouble(obj);
  double value_imag = PyComplex_ImagAsDouble(obj);
  if ((value_real == -1 || value_imag == -1) && PyErr_Occurred()) {
    throw std::exception();
  }
  *reinterpret_cast<dynd::complex<double> *>(out) = dynd::complex<double>(value_real, value_imag);
}

inline void convert_one_pyscalar_bytes(const ndt::type &tp, const char *arrmeta, char *out, PyObject *obj)
{
  dynd::bytes *out_asp = reinterpret_cast<dynd::bytes *>(out);
  char *data = NULL;
  intptr_t len = 0;
#if PY_VERSION_HEX >= 0x03000000
  if (PyBytes_Check(obj)) {
    if (PyBytes_AsStringAndSize(obj, &data, &len) < 0) {
#else
  if (PyString_Check(obj)) {
    if (PyString_AsStringAndSize(obj, &data, &len) < 0) {
#endif
      throw runtime_error("Error getting byte string data");
    }

    out_asp->assign(data, len);
  }
  else {
    throw dynd::type_error("wrong kind of string provided (require byte string "
                           "for dynd bytes type)");
  }
}

inline void convert_one_pyscalar_ustring(const ndt::type &tp, const char *arrmeta, char *out, PyObject *obj)
{
  dynd::string *out_usp = reinterpret_cast<dynd::string *>(out);
  if (PyUnicode_Check(obj)) {
    // Get it as UTF8
    py_ref utf8 = capture_if_not_null(PyUnicode_AsUTF8String(obj));
    char *s = NULL;
    Py_ssize_t len = 0;
    if (PyBytes_AsStringAndSize(utf8.get(), &s, &len) < 0) {
      throw exception();
    }
    out_usp->assign(s, len);
#if PY_VERSION_HEX < 0x03000000
  }
  else if (PyString_Check(obj)) {
    char *data = NULL;
    Py_ssize_t len = 0;
    if (PyString_AsStringAndSize(obj, &data, &len) < 0) {
      throw runtime_error("Error getting string data");
    }

    out_usp->resize(len);
    for (Py_ssize_t i = 0; i < len; ++i) {
      // Only let valid ascii get through
      if ((unsigned char)data[i] >= 128) {
        throw string_decode_error(data + i, data + i + 1, string_encoding_ascii);
      }
      out_usp->begin()[i] = data[i];
    }
#endif
  }
  else {
    throw dynd::type_error("wrong kind of string provided");
  }
}

inline void convert_one_pyscalar__type(const ndt::type &DYND_UNUSED(tp), const char *DYND_UNUSED(arrmeta), char *out,
                                       PyObject *obj)
{
  ndt::type obj_as_tp = dynd_ndt_cpp_type_for(obj);
  obj_as_tp.swap(*reinterpret_cast<ndt::type *>(out));
}

inline void convert_one_pyscalar_option(const ndt::type &tp, const char *arrmeta, char *out, PyObject *obj)
{
  if (obj == Py_None) {
    nd::old_assign_na(tp, arrmeta, out);
  }
  else {
    throw std::runtime_error("unable to convert to option value");
  }
}

template <convert_one_pyscalar_function_t ConvertOneFn>
static void fill_array_from_pylist(const ndt::type &tp, const char *arrmeta, char *data, PyObject *obj,
                                   const intptr_t *shape, size_t current_axis)
{
  if (shape[current_axis] == 0) {
    return;
  }

  Py_ssize_t size = PyList_GET_SIZE(obj);
  const char *element_arrmeta = arrmeta;
  ndt::type element_tp = tp.at_single(0, &element_arrmeta);
  if (shape[current_axis] >= 0) {
    // Fixed-sized dimension
    const fixed_dim_type_arrmeta *md = reinterpret_cast<const fixed_dim_type_arrmeta *>(arrmeta);
    intptr_t stride = md->stride;
    if (element_tp.is_scalar()) {
      for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(obj, i);
        ConvertOneFn(element_tp, element_arrmeta, data, item);
        data += stride;
      }
    }
    else {
      for (Py_ssize_t i = 0; i < size; ++i) {
        fill_array_from_pylist<ConvertOneFn>(element_tp, element_arrmeta, data, PyList_GET_ITEM(obj, i), shape,
                                             current_axis + 1);
        data += stride;
      }
    }
  }
  else {
    // Variable-sized dimension
    const ndt::var_dim_type::metadata_type *md = reinterpret_cast<const ndt::var_dim_type::metadata_type *>(arrmeta);
    intptr_t stride = md->stride;
    ndt::var_dim_type::data_type *out = reinterpret_cast<ndt::var_dim_type::data_type *>(data);
    char *out_end = NULL;

    out->begin = md->blockref->alloc(size);
    out_end = out->begin + size * stride;
    out->size = size;
    char *element_data = out->begin;
    if (element_tp.is_scalar()) {
      for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GET_ITEM(obj, i);
        ConvertOneFn(element_tp, element_arrmeta, element_data, item);
        element_data += stride;
      }
    }
    else {
      for (Py_ssize_t i = 0; i < size; ++i) {
        fill_array_from_pylist<ConvertOneFn>(element_tp, element_arrmeta, element_data, PyList_GET_ITEM(obj, i), shape,
                                             current_axis + 1);
        element_data += stride;
      }
    }
  }
}

static dynd::nd::array array_from_pylist(PyObject *obj)
{
  // TODO: Add ability to specify access flags (e.g. immutable)
  // Do a pass through all the data to deduce its type and shape
  vector<intptr_t> shape;
  ndt::type tp = ndt::make_type<void>();
  Py_ssize_t size = PyList_GET_SIZE(obj);
  shape.push_back(size);
  for (Py_ssize_t i = 0; i < size; ++i) {
    deduce_pylist_shape_and_dtype(PyList_GET_ITEM(obj, i), shape, tp, 1);
  }
  // If no type was deduced, return with no result. This will fall
  // through to the array_from_py_dynamic code.
  if (tp.get_id() == uninitialized_id || tp.get_id() == void_id) {
    return nd::array();
  }

  // Create the array
  nd::array result = pydynd::make_strided_array(tp, (int)shape.size(), &shape[0]);

  // Populate the array with data
  switch (tp.get_id()) {
  case bool_id:
    fill_array_from_pylist<convert_one_pyscalar_bool>(result.get_type(), result.get()->metadata(), result.data(), obj,
                                                      &shape[0], 0);
    break;
  case int32_id:
    fill_array_from_pylist<convert_one_pyscalar_int32>(result.get_type(), result.get()->metadata(), result.data(), obj,
                                                       &shape[0], 0);
    break;
  case int64_id:
    fill_array_from_pylist<convert_one_pyscalar_int64>(result.get_type(), result.get()->metadata(), result.data(), obj,
                                                       &shape[0], 0);
    break;
  case float32_id:
    fill_array_from_pylist<convert_one_pyscalar_float32>(result.get_type(), result.get()->metadata(), result.data(),
                                                         obj, &shape[0], 0);
    break;
  case float64_id:
    fill_array_from_pylist<convert_one_pyscalar_float64>(result.get_type(), result.get()->metadata(), result.data(),
                                                         obj, &shape[0], 0);
    break;
  case complex_float64_id:
    fill_array_from_pylist<convert_one_pyscalar_cdouble>(result.get_type(), result.get()->metadata(), result.data(),
                                                         obj, &shape[0], 0);
    break;
  case bytes_id:
    fill_array_from_pylist<convert_one_pyscalar_bytes>(result.get_type(), result.get()->metadata(), result.data(), obj,
                                                       &shape[0], 0);
    break;
  case string_id: {
    const ndt::base_string_type *ext = tp.extended<ndt::base_string_type>();
    if (ext->get_encoding() == string_encoding_utf_8) {
      fill_array_from_pylist<convert_one_pyscalar_ustring>(result.get_type(), result.get()->metadata(), result.data(),
                                                           obj, &shape[0], 0);
    }
    else {
      stringstream ss;
      ss << "Internal error: deduced type from Python list, " << tp << ", doesn't have a dynd array conversion";
      throw runtime_error(ss.str());
    }
    break;
  }
  case type_id: {
    fill_array_from_pylist<convert_one_pyscalar__type>(result.get_type(), result.get()->metadata(), result.data(), obj,
                                                       &shape[0], 0);
    break;
  }
  case option_id: {
    fill_array_from_pylist<convert_one_pyscalar_option>(result.get_type(), result.get()->metadata(), result.data(), obj,
                                                        &shape[0], 0);
    break;
  }
  default: {
    stringstream ss;
    ss << "Deduced type from Python list, " << tp << ", doesn't have a dynd array conversion function yet";
    throw runtime_error(ss.str());
  }
  }
  result.get_type().extended()->arrmeta_finalize_buffers(result.get()->metadata());
  return result;
}

dynd::nd::array pydynd::array_from_py(PyObject *obj, uint32_t access_flags, bool always_copy)
{
  // If it's a Cython w_array
  if (PyObject_TypeCheck(obj, get_array_pytypeobject())) {
    const nd::array &result = pydynd::array_to_cpp_ref(obj);
    if (always_copy) {
      return result.eval_copy(access_flags);
    }
    else {
      if (access_flags != 0) {
        uint64_t raf = result.get_flags();
        if ((access_flags & nd::immutable_access_flag) && !(raf & nd::immutable_access_flag)) {
          throw runtime_error("cannot view a non-immutable dynd array as immutable");
        }
        if ((access_flags & nd::write_access_flag) && !(raf & nd::write_access_flag)) {
          throw runtime_error("cannot view a readonly dynd array as readwrite");
        }
      }
      return result;
    }
  }

#if DYND_NUMPY_INTEROP
  if (PyArray_Check(obj)) {
    return array_from_numpy_array((PyArrayObject *)obj, access_flags, always_copy);
  }
  else if (PyArray_IsScalar(obj, Generic)) {
    return array_from_numpy_scalar(obj, access_flags);
  }
#endif // DYND_NUMPY_INTEROP

  nd::array result;

  if (PyBool_Check(obj)) {
    result = nd::array(obj == Py_True);
#if PY_VERSION_HEX < 0x03000000
  }
  else if (PyInt_Check(obj)) {
    long value = PyInt_AS_LONG(obj);
#if SIZEOF_LONG > SIZEOF_INT
    // Use a 32-bit int if it fits.
    if (value >= INT_MIN && value <= INT_MAX) {
      result = nd::array(static_cast<int>(value));
    }
    else {
      result = nd::array(value);
    }
#else
    result = nd::array(value);
#endif
#endif // PY_VERSION_HEX < 0x03000000
  }
  else if (PyLong_Check(obj)) {
    PY_LONG_LONG value = PyLong_AsLongLong(obj);
    if (value == -1 && PyErr_Occurred()) {
      throw runtime_error("error converting int value");
    }

    // Use a 32-bit int if it fits.
    if (value >= INT_MIN && value <= INT_MAX) {
      result = nd::array(static_cast<int>(value));
    }
    else {
      result = nd::array(value);
    }
  }
  else if (PyFloat_Check(obj)) {
    result = nd::array(PyFloat_AS_DOUBLE(obj));
  }
  else if (PyComplex_Check(obj)) {
    result = nd::array(dynd::complex<double>(PyComplex_RealAsDouble(obj), PyComplex_ImagAsDouble(obj)));
#if PY_VERSION_HEX < 0x03000000
  }
  else if (PyString_Check(obj)) {
    char *data = NULL;
    Py_ssize_t len = 0;
    if (PyString_AsStringAndSize(obj, &data, &len) < 0) {
      throw runtime_error("Error getting string data");
    }

    for (Py_ssize_t i = 0; i < len; ++i) {
      // Only let valid ascii get through
      if ((unsigned char)data[i] >= 128) {
        throw string_decode_error(data + i, data + i + 1, string_encoding_ascii);
      }
    }
    result = nd::empty(ndt::make_type<ndt::string_type>());
    reinterpret_cast<dynd::string *>(result.data())->assign(data, len);
#else
  }
  else if (PyBytes_Check(obj)) {
    // Cannot provide write access unless a copy is being made
    if ((access_flags & nd::write_access_flag) != 0) {
      if (always_copy) {
        // If a readwrite copy is requested, make a new bytes array and copy the
        // data.
        // For readonly copies, no need to copy because the data is immutable.
        char *data = NULL;
        intptr_t len = 0;
        if (PyBytes_AsStringAndSize(obj, &data, &len) < 0) {
          throw runtime_error("Error getting byte string data");
        }
        result = nd::empty(ndt::make_type<ndt::bytes_type>());
        reinterpret_cast<bytes *>(result.data())->assign(data, len);
        return result;
      }
      else {
        throw runtime_error("cannot create a writable view of a python bytes object");
      }
    }

    char *data = NULL;
    intptr_t len = 0;
    if (PyBytes_AsStringAndSize(obj, &data, &len) < 0) {
      throw runtime_error("Error getting byte string data");
    }
    ndt::type d = ndt::make_type<ndt::bytes_type>(1);
    // Python bytes are immutable, so simply use the existing memory with an
    // external memory
    Py_INCREF(obj);
    nd::memory_block bytesref =
        nd::make_memory_block<nd::external_memory_block>(reinterpret_cast<void *>(obj), &py_decref_function);
    result = nd::empty(d);
    // The scalar consists of pointers to the byte string data
    reinterpret_cast<dynd::string *>(result.data())->assign(data, len);
    return result;
#endif
  }
  else if (PyUnicode_Check(obj)) {
    py_ref utf8 = capture_if_not_null(PyUnicode_AsUTF8String(obj));
    char *s = NULL;
    Py_ssize_t len = 0;
    if (PyBytes_AsStringAndSize(utf8.get(), &s, &len) < 0) {
      throw exception();
    }
    result = nd::empty(ndt::make_type<ndt::string_type>());
    reinterpret_cast<dynd::string *>(result.data())->assign(s, len);
  }
  else if (PyObject_TypeCheck(obj, get_type_pytypeobject())) {
    result = nd::array(type_to_cpp_ref(obj));
  }
  else if (PyList_Check(obj)) {
    result = array_from_pylist(obj);
  }
  else if (PyType_Check(obj)) {
    result = nd::array(dynd_ndt_cpp_type_for(obj));
#if DYND_NUMPY_INTEROP
  }
  else if (PyArray_DescrCheck(obj)) {
    result = nd::array(dynd_ndt_cpp_type_for(obj));
#endif // DYND_NUMPY_INTEROP
  }

  if (result.get() == NULL) {
    py_ref pytpstr = capture_if_not_null(PyObject_Str((PyObject *)Py_TYPE(obj)));
    stringstream ss;
    ss << "could not convert python object of type ";
    ss << pystring_as_string(pytpstr.get());
    ss << " into a dynd array";
    throw std::runtime_error(ss.str());
  }

  return result;
}

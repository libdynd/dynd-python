//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>
#include <datetime.h>

#include <dynd/callable.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/time_type.hpp>
#include <dynd/types/datetime_type.hpp>
#include <dynd/types/type_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/substitute_shape.hpp>
#include <dynd/memblock/external_memory_block.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/type_promotion.hpp>
#include <dynd/exceptions.hpp>

#include "array_from_py.hpp"
#include "array_from_py_typededuction.hpp"
#include "array_functions.hpp"
#include "type_functions.hpp"
#include "utility_functions.hpp"
#include "numpy_interop.hpp"
#include "types/pyobject_type.hpp"
#include "conversions.hpp"

using namespace std;
using namespace dynd;
using namespace pydynd;

void pydynd::init_array_from_py()
{
  // Initialize the pydatetime API
  PyDateTime_IMPORT;
}

typedef void (*convert_one_pyscalar_function_t)(const ndt::type &tp,
                                                const char *arrmeta, char *out,
                                                PyObject *obj);

inline void convert_one_pyscalar_bool(const ndt::type &tp, const char *arrmeta,
                                      char *out, PyObject *obj)
{
  *out = (PyObject_IsTrue(obj) != 0);
}

inline void convert_one_pyscalar_int32(const ndt::type &tp, const char *arrmeta,
                                       char *out, PyObject *obj)
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

inline void convert_one_pyscalar_int64(const ndt::type &tp, const char *arrmeta,
                                       char *out, PyObject *obj)
{
  int64_t value = PyLong_AsLongLong(obj);
  if (value == -1 && PyErr_Occurred()) {
    throw std::exception();
  }
  *reinterpret_cast<int64_t *>(out) = value;
}

inline void convert_one_pyscalar_float32(const ndt::type &tp,
                                         const char *arrmeta, char *out,
                                         PyObject *obj)
{
  double value = PyFloat_AsDouble(obj);
  if (value == -1 && PyErr_Occurred()) {
    throw std::exception();
  }
  *reinterpret_cast<float *>(out) = static_cast<float>(value);
}

inline void convert_one_pyscalar_float64(const ndt::type &tp,
                                         const char *arrmeta, char *out,
                                         PyObject *obj)
{
  double value = PyFloat_AsDouble(obj);
  if (value == -1 && PyErr_Occurred()) {
    throw std::exception();
  }
  *reinterpret_cast<double *>(out) = value;
}

inline void convert_one_pyscalar_cdouble(const ndt::type &tp,
                                         const char *arrmeta, char *out,
                                         PyObject *obj)
{
  double value_real = PyComplex_RealAsDouble(obj);
  double value_imag = PyComplex_ImagAsDouble(obj);
  if ((value_real == -1 || value_imag == -1) && PyErr_Occurred()) {
    throw std::exception();
  }
  *reinterpret_cast<dynd::complex<double> *>(out) =
      dynd::complex<double>(value_real, value_imag);
}

inline void convert_one_pyscalar_bytes(const ndt::type &tp, const char *arrmeta,
                                       char *out, PyObject *obj)
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

inline void convert_one_pyscalar_ustring(const ndt::type &tp,
                                         const char *arrmeta, char *out,
                                         PyObject *obj)
{
  dynd::string *out_usp = reinterpret_cast<dynd::string *>(out);
  if (PyUnicode_Check(obj)) {
    // Get it as UTF8
    pyobject_ownref utf8(PyUnicode_AsUTF8String(obj));
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
        throw string_decode_error(data + i, data + i + 1,
                                  string_encoding_ascii);
      }
      out_usp->begin()[i] = data[i];
    }
#endif
  }
  else {
    throw dynd::type_error("wrong kind of string provided");
  }
}

inline void convert_one_pyscalar_date(const ndt::type &tp, const char *arrmeta,
                                      char *out, PyObject *obj)
{
  if (!PyDate_Check(obj)) {
    throw dynd::type_error("input object is not a date as expected");
  }
  const ndt::date_type *dd = tp.extended<ndt::date_type>();
  dd->set_ymd(arrmeta, out, assign_error_fractional, PyDateTime_GET_YEAR(obj),
              PyDateTime_GET_MONTH(obj), PyDateTime_GET_DAY(obj));
}

inline void convert_one_pyscalar_time(const ndt::type &tp, const char *arrmeta,
                                      char *out, PyObject *obj)
{
  if (!PyTime_Check(obj)) {
    throw dynd::type_error("input object is not a time as expected");
  }
  const ndt::time_type *tt = tp.extended<ndt::time_type>();
  tt->set_time(
      arrmeta, out, assign_error_fractional, PyDateTime_TIME_GET_HOUR(obj),
      PyDateTime_TIME_GET_MINUTE(obj), PyDateTime_TIME_GET_SECOND(obj),
      PyDateTime_TIME_GET_MICROSECOND(obj) * DYND_TICKS_PER_MICROSECOND);
}

inline void convert_one_pyscalar_datetime(const ndt::type &tp,
                                          const char *arrmeta, char *out,
                                          PyObject *obj)
{
  if (!PyDateTime_Check(obj)) {
    throw dynd::type_error("input object is not a datetime as expected");
  }
  if (((PyDateTime_DateTime *)obj)->hastzinfo &&
      ((PyDateTime_DateTime *)obj)->tzinfo != NULL) {
    throw runtime_error("Converting datetimes with a timezone to dynd arrays "
                        "is not yet supported");
  }
  const ndt::datetime_type *dd = tp.extended<ndt::datetime_type>();
  dd->set_cal(arrmeta, out, assign_error_fractional, PyDateTime_GET_YEAR(obj),
              PyDateTime_GET_MONTH(obj), PyDateTime_GET_DAY(obj),
              PyDateTime_DATE_GET_HOUR(obj), PyDateTime_DATE_GET_MINUTE(obj),
              PyDateTime_DATE_GET_SECOND(obj),
              PyDateTime_DATE_GET_MICROSECOND(obj) * 10);
}

inline void convert_one_pyscalar__type(const ndt::type &DYND_UNUSED(tp),
                                       const char *DYND_UNUSED(arrmeta),
                                       char *out, PyObject *obj)
{
  ndt::type obj_as_tp = make__type_from_pyobject(obj);
  obj_as_tp.swap(*reinterpret_cast<ndt::type *>(out));
}

inline void convert_one_pyscalar_option(const ndt::type &tp,
                                        const char *arrmeta, char *out,
                                        PyObject *obj)
{
  if (obj == Py_None) {
    tp.extended<ndt::option_type>()->assign_na(arrmeta, out,
                                               &eval::default_eval_context);
  }
  else {
    throw std::runtime_error("unable to convert to option value");
  }
}

template <convert_one_pyscalar_function_t ConvertOneFn>
static void fill_array_from_pylist(const ndt::type &tp, const char *arrmeta,
                                   char *data, PyObject *obj,
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
    const fixed_dim_type_arrmeta *md =
        reinterpret_cast<const fixed_dim_type_arrmeta *>(arrmeta);
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
        fill_array_from_pylist<ConvertOneFn>(element_tp, element_arrmeta, data,
                                             PyList_GET_ITEM(obj, i), shape,
                                             current_axis + 1);
        data += stride;
      }
    }
  }
  else {
    // Variable-sized dimension
    const ndt::var_dim_type::metadata_type *md =
        reinterpret_cast<const ndt::var_dim_type::metadata_type *>(arrmeta);
    intptr_t stride = md->stride;
    ndt::var_dim_type::data_type *out =
        reinterpret_cast<ndt::var_dim_type::data_type *>(data);
    char *out_end = NULL;

    memory_block_data::api *allocator = md->blockref->get_api();
    out->begin = allocator->allocate(md->blockref.get(), size);
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
        fill_array_from_pylist<ConvertOneFn>(
            element_tp, element_arrmeta, element_data, PyList_GET_ITEM(obj, i),
            shape, current_axis + 1);
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
  ndt::type tp(void_id);
  Py_ssize_t size = PyList_GET_SIZE(obj);
  shape.push_back(size);
  for (Py_ssize_t i = 0; i < size; ++i) {
    deduce_pylist_shape_and_dtype(PyList_GET_ITEM(obj, i), shape, tp, 1);
  }
  // If no type was deduced, return with no result. This will fall
  // through to the array_from_py_dynamic code.
  if (tp.get_id() == uninitialized_id ||
      tp.get_id() == void_id) {
    return nd::array();
  }

  // Create the array
  nd::array result =
      pydynd::make_strided_array(tp, (int)shape.size(), &shape[0]);

  // Populate the array with data
  switch (tp.get_id()) {
  case bool_id:
    fill_array_from_pylist<convert_one_pyscalar_bool>(
        result.get_type(), result.get()->metadata(), result.data(), obj,
        &shape[0], 0);
    break;
  case int32_id:
    fill_array_from_pylist<convert_one_pyscalar_int32>(
        result.get_type(), result.get()->metadata(), result.data(), obj,
        &shape[0], 0);
    break;
  case int64_id:
    fill_array_from_pylist<convert_one_pyscalar_int64>(
        result.get_type(), result.get()->metadata(), result.data(), obj,
        &shape[0], 0);
    break;
  case float32_id:
    fill_array_from_pylist<convert_one_pyscalar_float32>(
        result.get_type(), result.get()->metadata(), result.data(), obj,
        &shape[0], 0);
    break;
  case float64_id:
    fill_array_from_pylist<convert_one_pyscalar_float64>(
        result.get_type(), result.get()->metadata(), result.data(), obj,
        &shape[0], 0);
    break;
  case complex_float64_id:
    fill_array_from_pylist<convert_one_pyscalar_cdouble>(
        result.get_type(), result.get()->metadata(), result.data(), obj,
        &shape[0], 0);
    break;
  case bytes_id:
    fill_array_from_pylist<convert_one_pyscalar_bytes>(
        result.get_type(), result.get()->metadata(), result.data(), obj,
        &shape[0], 0);
    break;
  case string_id: {
    const ndt::base_string_type *ext = tp.extended<ndt::base_string_type>();
    if (ext->get_encoding() == string_encoding_utf_8) {
      fill_array_from_pylist<convert_one_pyscalar_ustring>(
          result.get_type(), result.get()->metadata(), result.data(), obj,
          &shape[0], 0);
    }
    else {
      stringstream ss;
      ss << "Internal error: deduced type from Python list, " << tp
         << ", doesn't have a dynd array conversion";
      throw runtime_error(ss.str());
    }
    break;
  }
  case date_id: {
    fill_array_from_pylist<convert_one_pyscalar_date>(
        result.get_type(), result.get()->metadata(), result.data(), obj,
        &shape[0], 0);
    break;
  }
  case time_id: {
    fill_array_from_pylist<convert_one_pyscalar_time>(
        result.get_type(), result.get()->metadata(), result.data(), obj,
        &shape[0], 0);
    break;
  }
  case datetime_id: {
    fill_array_from_pylist<convert_one_pyscalar_datetime>(
        result.get_type(), result.get()->metadata(), result.data(), obj,
        &shape[0], 0);
    break;
  }
  case type_id: {
    fill_array_from_pylist<convert_one_pyscalar__type>(
        result.get_type(), result.get()->metadata(), result.data(), obj,
        &shape[0], 0);
    break;
  }
  case option_id: {
    fill_array_from_pylist<convert_one_pyscalar_option>(
        result.get_type(), result.get()->metadata(), result.data(), obj,
        &shape[0], 0);
    break;
  }
  default: {
    stringstream ss;
    ss << "Deduced type from Python list, " << tp
       << ", doesn't have a dynd array conversion function yet";
    throw runtime_error(ss.str());
  }
  }
  result.get_type().extended()->arrmeta_finalize_buffers(
      result.get()->metadata());
  return result;
}

dynd::nd::array pydynd::array_from_py(PyObject *obj, uint32_t access_flags,
                                      bool always_copy)
{
  // If it's a Cython w_array
  if (PyObject_TypeCheck(obj, get_array_pytypeobject())) {
    const nd::array &result = pydynd::array_to_cpp_ref(obj);
    if (always_copy) {
      return result.eval_copy(access_flags);
    }
    else {
      if (access_flags != 0) {
        uint32_t raf = result.get_access_flags();
        if ((access_flags & nd::immutable_access_flag) &&
            !(raf & nd::immutable_access_flag)) {
          throw runtime_error(
              "cannot view a non-immutable dynd array as immutable");
        }
        if ((access_flags & nd::write_access_flag) &&
            !(raf & nd::write_access_flag)) {
          throw runtime_error("cannot view a readonly dynd array as readwrite");
        }
      }
      return result;
    }
  }

#if DYND_NUMPY_INTEROP
  if (PyArray_Check(obj)) {
    return array_from_numpy_array((PyArrayObject *)obj, access_flags,
                                  always_copy);
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
    result = nd::array(dynd::complex<double>(PyComplex_RealAsDouble(obj),
                                             PyComplex_ImagAsDouble(obj)));
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
        throw string_decode_error(data + i, data + i + 1,
                                  string_encoding_ascii);
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
        result = nd::empty(ndt::bytes_type::make());
        reinterpret_cast<bytes *>(result.data())->assign(data, len);
        result.get()->flags = access_flags;
        return result;
      }
      else {
        throw runtime_error(
            "cannot create a writable view of a python bytes object");
      }
    }

    char *data = NULL;
    intptr_t len = 0;
    if (PyBytes_AsStringAndSize(obj, &data, &len) < 0) {
      throw runtime_error("Error getting byte string data");
    }
    ndt::type d = ndt::bytes_type::make(1);
    // Python bytes are immutable, so simply use the existing memory with an
    // external memory
    Py_INCREF(obj);
    intrusive_ptr<memory_block_data> bytesref = make_external_memory_block(
        reinterpret_cast<void *>(obj), &py_decref_function);
    char *data_ptr;
    result =
        nd::array(reinterpret_cast<dynd::array_preamble *>(
                      make_array_memory_block(d.extended()->get_arrmeta_size(),
                                              d.get_data_size(),
                                              d.get_data_alignment(), &data_ptr)
                          .get()),
                  true);
    result.get()->data = data_ptr;
    result.get()->owner = NULL;
    result.get()->tp = d;
    // The scalar consists of pointers to the byte string data
    reinterpret_cast<dynd::string *>(data_ptr)->assign(data, len);
    // The arrmeta
    result.get()->flags = nd::immutable_access_flag | nd::read_access_flag;
    // Because this is a view into another object's memory, skip the later
    // processing
    return result;
#endif
  }
  else if (PyUnicode_Check(obj)) {
    pyobject_ownref utf8(PyUnicode_AsUTF8String(obj));
    char *s = NULL;
    Py_ssize_t len = 0;
    if (PyBytes_AsStringAndSize(utf8.get(), &s, &len) < 0) {
      throw exception();
    }
    result = nd::empty(ndt::make_type<ndt::string_type>());
    reinterpret_cast<dynd::string *>(result.data())->assign(s, len);
  }
  else if (PyDateTime_Check(obj)) {
    if (((PyDateTime_DateTime *)obj)->hastzinfo &&
        ((PyDateTime_DateTime *)obj)->tzinfo != NULL) {
      throw runtime_error("Converting datetimes with a timezone to dynd "
                          "arrays is not yet supported");
    }
    ndt::type d = ndt::datetime_type::make();
    const ndt::datetime_type *dd = d.extended<ndt::datetime_type>();
    result = nd::empty(d);
    dd->set_cal(result.get()->metadata(), result.get()->data,
                assign_error_fractional, PyDateTime_GET_YEAR(obj),
                PyDateTime_GET_MONTH(obj), PyDateTime_GET_DAY(obj),
                PyDateTime_DATE_GET_HOUR(obj), PyDateTime_DATE_GET_MINUTE(obj),
                PyDateTime_DATE_GET_SECOND(obj),
                PyDateTime_DATE_GET_MICROSECOND(obj) * 10);
  }
  else if (PyDate_Check(obj)) {
    ndt::type d = ndt::date_type::make();
    const ndt::date_type *dd = d.extended<ndt::date_type>();
    result = nd::empty(d);
    dd->set_ymd(result.get()->metadata(), result.get()->data,
                assign_error_fractional, PyDateTime_GET_YEAR(obj),
                PyDateTime_GET_MONTH(obj), PyDateTime_GET_DAY(obj));
  }
  else if (PyTime_Check(obj)) {
    if (((PyDateTime_DateTime *)obj)->hastzinfo &&
        ((PyDateTime_DateTime *)obj)->tzinfo != NULL) {
      throw runtime_error("Converting times with a timezone to dynd "
                          "arrays is not yet supported");
    }
    ndt::type d = ndt::time_type::make(tz_abstract);
    const ndt::time_type *tt = d.extended<ndt::time_type>();
    result = nd::empty(d);
    tt->set_time(result.get()->metadata(), result.get()->data,
                 assign_error_fractional, PyDateTime_TIME_GET_HOUR(obj),
                 PyDateTime_TIME_GET_MINUTE(obj),
                 PyDateTime_TIME_GET_SECOND(obj),
                 PyDateTime_TIME_GET_MICROSECOND(obj) * 10);
  }
  else if (PyObject_TypeCheck(obj, get_type_pytypeobject())) {
    result = nd::array(type_to_cpp_ref(obj));
  }
  else if (PyList_Check(obj)) {
    result = array_from_pylist(obj);
  }
  else if (PyType_Check(obj)) {
    result = nd::array(make__type_from_pyobject(obj));
#if DYND_NUMPY_INTEROP
  }
  else if (PyArray_DescrCheck(obj)) {
    result = nd::array(make__type_from_pyobject(obj));
#endif // DYND_NUMPY_INTEROP
  }

  if (result.get() == NULL) {
    pyobject_ownref pytpstr(PyObject_Str((PyObject *)Py_TYPE(obj)));
    stringstream ss;
    ss << "could not convert python object of type ";
    ss << pystring_as_string(pytpstr.get());
    ss << " into a dynd array";
    throw std::runtime_error(ss.str());
  }

  // If write access wasn't specified, we can flag it as
  // immutable, because it's a newly allocated object.
  if (access_flags != 0 && (access_flags & nd::write_access_flag) == 0) {
    result.flag_as_immutable();
  }

  return result;
}

dynd::ndt::type pydynd::xtype_for_prefix(PyObject *obj)
{
  // If it's a Cython w_array
  if (PyObject_TypeCheck(obj, get_array_pytypeobject())) {
    return pydynd::array_to_cpp_ref(obj).get_type();
  }

#if DYND_NUMPY_INTEROP
  if (PyArray_Check(obj)) {
    return array_from_numpy_array2((PyArrayObject *)obj);
  }

#endif // DYND_NUMPY_INTEROP
  if (PyBool_Check(obj)) {
    return ndt::make_type<bool>();
  }
#if PY_VERSION_HEX < 0x03000000
  if (PyInt_Check(obj)) {
    long value = PyInt_AS_LONG(obj);
#if SIZEOF_LONG > SIZEOF_INT
    // Use a 32-bit int if it fits.
    if (value >= INT_MIN && value <= INT_MAX) {
      return ndt::make_type<int>();
    }
    else {
      return ndt::make_type<long>();
    }
#else
    return ndt::make_type<long>();
#endif
  }
#endif // PY_VERSION_HEX < 0x03000000
  if (PyLong_Check(obj)) {
    PY_LONG_LONG value = PyLong_AsLongLong(obj);
    if (value == -1 && PyErr_Occurred()) {
      throw runtime_error("error converting int value");
    }

    // Use a 32-bit int if it fits.
    if (value >= INT_MIN && value <= INT_MAX) {
      return ndt::make_type<int>();
    }
    else {
      return ndt::make_type<PY_LONG_LONG>();
    }
  }

  return dynd::ndt::type();
}

dynd::ndt::type pydynd::xarray_from_pylist(PyObject *obj)
{
  // TODO: Add ability to specify access flags (e.g. immutable)
  // Do a pass through all the data to deduce its type and shape
  vector<intptr_t> shape;
  ndt::type tp(void_id);
  Py_ssize_t size = PyList_GET_SIZE(obj);
  shape.push_back(size);
  for (Py_ssize_t i = 0; i < size; ++i) {
    deduce_pylist_shape_and_dtype(PyList_GET_ITEM(obj, i), shape, tp, 1);
  }

  if (tp.get_id() == void_id) {
    tp = dynd::ndt::type(int32_id);
  }

  return dynd::ndt::make_type(shape.size(), shape.data(), tp);
}

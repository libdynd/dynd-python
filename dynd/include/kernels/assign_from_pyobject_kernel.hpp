//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/assignment.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/option.hpp>
#include <dynd/parse.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/categorical_type.hpp>
#include <dynd/types/fixed_bytes_kind_type.hpp>
#include <dynd/types/fixed_bytes_type.hpp>
#include <dynd/types/fixed_string_kind_type.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/var_dim_type.hpp>

#include "array_conversions.hpp"
#include "array_functions.hpp"
#include "copy_from_numpy_arrfunc.hpp"
#include "type_deduction.hpp"
#include "type_functions.hpp"
#include "types/pyobject_type.hpp"

using namespace dynd;

namespace pydynd {
namespace nd {

  inline void typed_data_assign(const dynd::ndt::type &dst_tp, const char *dst_arrmeta, char *dst_data,
                                const dynd::ndt::type &src_tp, const char *src_arrmeta, const char *src_data)
  {
    dynd::nd::array kwd = dynd::nd::empty(dynd::ndt::make_type<dynd::ndt::option_type>(dynd::ndt::make_type<int>()));
    *reinterpret_cast<int *>(kwd.data()) = static_cast<int>(dynd::assign_error_fractional);
    std::map<std::string, dynd::ndt::type> tp_vars;
    dynd::nd::assign->call(dst_tp, dst_arrmeta, dst_data, 1, &src_tp, &src_arrmeta,
                           const_cast<char *const *>(&src_data), 1, &kwd, tp_vars);
  }

  inline void typed_data_assign(const dynd::ndt::type &dst_tp, const char *dst_arrmeta, char *dst_data,
                                const dynd::nd::array &src_arr)
  {
    pydynd::nd::typed_data_assign(dst_tp, dst_arrmeta, dst_data, src_arr.get_type(), src_arr.get()->metadata(),
                                  src_arr.cdata());
  }

} // namespace pydynd::nd
} // namespace pydynd

namespace {

template <typename ReturnType, typename Enable = void>
struct assign_from_pyobject_kernel;

template <>
struct assign_from_pyobject_kernel<bool> : nd::base_strided_kernel<assign_from_pyobject_kernel<bool>, 1> {
  void single(char *dst, char *const *src)
  {
    PyObject *src_obj = *reinterpret_cast<PyObject **>(src[0]);
    if (src_obj == Py_True) {
      *dst = 1;
    }
    else if (src_obj == Py_False) {
      *dst = 0;
    }
    else {
      *dst = pydynd::array_from_py(src_obj, 0, false).as<dynd::bool1>();
    }
  }
};

void pyint_to_int(int8_t *out, PyObject *obj)
{
  long v = PyLong_AsLong(obj);
  if (v == -1 && PyErr_Occurred()) {
    throw std::exception();
  }
  //    if (dynd::overflow_check<int8_t>::is_overflow(v, true)) {
  //    throw std::overflow_error("overflow assigning to dynd int8");
  //}
  *out = static_cast<int8_t>(v);
}

void pyint_to_int(int16_t *out, PyObject *obj)
{
  long v = PyLong_AsLong(obj);
  if (v == -1 && PyErr_Occurred()) {
    throw std::exception();
  }
  //    if (dynd::overflow_check<int16_t>::is_overflow(v, true)) {
  //    throw std::overflow_error("overflow assigning to dynd int16");
  //}
  *out = static_cast<int16_t>(v);
}

void pyint_to_int(int32_t *out, PyObject *obj)
{
  long v = PyLong_AsLong(obj);
  if (v == -1 && PyErr_Occurred()) {
    throw std::exception();
  }
  //    if (dynd::overflow_check<int32_t>::is_overflow(v, true)) {
  //    throw std::overflow_error("overflow assigning to dynd int32");
  //}
  *out = static_cast<int32_t>(v);
}

void pyint_to_int(int64_t *out, PyObject *obj)
{
  int64_t v = PyLong_AsLongLong(obj);
  if (v == -1 && PyErr_Occurred()) {
    throw std::exception();
  }
  *out = static_cast<int64_t>(v);
}

void pyint_to_int(dynd::int128 *out, PyObject *obj)
{
#if PY_VERSION_HEX < 0x03000000
  if (PyInt_Check(obj)) {
    long value = PyInt_AS_LONG(obj);
    *out = value;
    return;
  }
#endif
  uint64_t lo = PyLong_AsUnsignedLongLongMask(obj);
  pydynd::py_ref sixtyfour = pydynd::capture_if_not_null(PyLong_FromLong(64));
  pydynd::py_ref value_shr1 = pydynd::capture_if_not_null(PyNumber_Rshift(obj, sixtyfour.get()));
  uint64_t hi = PyLong_AsUnsignedLongLongMask(value_shr1.get());
  dynd::int128 result(hi, lo);

  // Shift right another 64 bits, and check that nothing is remaining
  pydynd::py_ref value_shr2 = pydynd::capture_if_not_null(PyNumber_Rshift(value_shr1.get(), sixtyfour.get()));
  long remaining = PyLong_AsLong(value_shr2.get());
  if ((remaining != 0 || (remaining == 0 && result.is_negative())) &&
      (remaining != -1 || PyErr_Occurred() || (remaining == -1 && !result.is_negative()))) {
    throw std::overflow_error("int is too big to fit in an int128");
  }

  *out = result;
}

void pyint_to_int(uint8_t *out, PyObject *obj)
{
  unsigned long v = PyLong_AsUnsignedLong(obj);
  if (v == -1 && PyErr_Occurred()) {
    throw std::exception();
  }
  if (dynd::is_overflow<uint8_t>(v)) {
    throw std::overflow_error("overflow assigning to dynd uint8");
  }
  *out = static_cast<uint8_t>(v);
}

void pyint_to_int(uint16_t *out, PyObject *obj)
{
  unsigned long v = PyLong_AsUnsignedLong(obj);
  if (v == -1 && PyErr_Occurred()) {
    throw std::exception();
  }
  if (dynd::is_overflow<uint16_t>(v)) {
    throw std::overflow_error("overflow assigning to dynd uint16");
  }
  *out = static_cast<uint16_t>(v);
}

void pyint_to_int(uint32_t *out, PyObject *obj)
{
  unsigned long v = PyLong_AsUnsignedLong(obj);
  if (v == -1 && PyErr_Occurred()) {
    throw std::exception();
  }
  if (dynd::is_overflow<uint32_t>(v)) {
    throw std::overflow_error("overflow assigning to dynd uint32");
  }
  *out = static_cast<uint32_t>(v);
}

void pyint_to_int(uint64_t *out, PyObject *obj)
{
#if PY_VERSION_HEX < 0x03000000
  if (PyInt_Check(obj)) {
    long value = PyInt_AS_LONG(obj);
    if (value < 0) {
      throw std::overflow_error("overflow assigning to dynd uint64");
    }
    *out = static_cast<unsigned long>(value);
    return;
  }
#endif
  uint64_t v = PyLong_AsUnsignedLongLong(obj);
  if (v == -1 && PyErr_Occurred()) {
    throw std::exception();
  }
  *out = v;
}

void pyint_to_int(dynd::uint128 *out, PyObject *obj)
{
#if PY_VERSION_HEX < 0x03000000
  if (PyInt_Check(obj)) {
    long value = PyInt_AS_LONG(obj);
    if (value < 0) {
      throw std::overflow_error("overflow assigning to dynd uint128");
    }
    *out = static_cast<unsigned long>(value);
    return;
  }
#endif
  uint64_t lo = PyLong_AsUnsignedLongLongMask(obj);
  pydynd::py_ref sixtyfour = pydynd::capture_if_not_null(PyLong_FromLong(64));
  pydynd::py_ref value_shr1 = pydynd::capture_if_not_null(PyNumber_Rshift(obj, sixtyfour.get()));
  uint64_t hi = PyLong_AsUnsignedLongLongMask(value_shr1.get());
  dynd::uint128 result(hi, lo);

  // Shift right another 64 bits, and check that nothing is remaining
  pydynd::py_ref value_shr2 = pydynd::capture_if_not_null(PyNumber_Rshift(value_shr1.get(), sixtyfour.get()));
  long remaining = PyLong_AsLong(value_shr2.get());
  if (remaining != 0) {
    throw std::overflow_error("int is too big to fit in an uint128");
  }

  *out = result;
}

template <typename ReturnType>
struct assign_from_pyobject_kernel<ReturnType, std::enable_if_t<is_signed_integral<ReturnType>::value>>
    : dynd::nd::base_strided_kernel<assign_from_pyobject_kernel<ReturnType>, 1> {
  void single(char *dst, char *const *src)
  {
    PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);
    if (PyLong_Check(src_obj)
#if PY_VERSION_HEX < 0x03000000
        || PyInt_Check(src_obj)
#endif
            ) {
      pyint_to_int(reinterpret_cast<ReturnType *>(dst), src_obj);
    }
#if DYND_NUMPY_INTEROP
    else if (PyArray_Check(src_obj)) {
      *reinterpret_cast<ReturnType *>(dst) =
          pydynd::array_from_numpy_array((PyArrayObject *)src_obj, 0, true).as<ReturnType>();
    }
    else if (PyArray_IsScalar(src_obj, Generic)) {
      *reinterpret_cast<ReturnType *>(dst) = pydynd::array_from_numpy_scalar(src_obj, 0).as<ReturnType>();
    }
#endif
    else {
      int overflow;
      long value = PyLong_AsLongAndOverflow(src_obj, &overflow);
      if (overflow == 0 && value == -1) {
        throw std::runtime_error("cannot assign Python object to integer");
      }

      *reinterpret_cast<ReturnType *>(dst) = static_cast<ReturnType>(value);
    }
  }
};

template <typename ReturnType>
struct assign_from_pyobject_kernel<ReturnType, std::enable_if_t<is_unsigned_integral<ReturnType>::value>>
    : dynd::nd::base_strided_kernel<assign_from_pyobject_kernel<ReturnType>, 1> {
  void single(char *dst, char *const *src)
  {
    PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);
    if (PyLong_Check(src_obj)
#if PY_VERSION_HEX < 0x03000000
        || PyInt_Check(src_obj)
#endif
            ) {
      pyint_to_int(reinterpret_cast<ReturnType *>(dst), src_obj);
    }
    else {
      *reinterpret_cast<ReturnType *>(dst) = pydynd::array_from_py(src_obj, 0, false).as<ReturnType>();
    }
  }
};

template <typename ReturnType>
struct assign_from_pyobject_kernel<ReturnType, std::enable_if_t<is_floating_point<ReturnType>::value>>
    : nd::base_strided_kernel<assign_from_pyobject_kernel<ReturnType>, 1> {
  void single(char *dst, char *const *src)
  {
    PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);
    if (PyFloat_Check(src_obj)) {
      double v = PyFloat_AsDouble(src_obj);
      if (v == -1 && PyErr_Occurred()) {
        throw std::exception();
      }
      *reinterpret_cast<ReturnType *>(dst) = static_cast<ReturnType>(v);
    }
    else {
      *reinterpret_cast<ReturnType *>(dst) = pydynd::array_from_py(src_obj, 0, false).as<ReturnType>();
    }
  }
};

template <typename ReturnType>
struct assign_from_pyobject_kernel<ReturnType, std::enable_if_t<is_complex<ReturnType>::value>>
    : nd::base_strided_kernel<assign_from_pyobject_kernel<ReturnType>, 1> {
  typedef ReturnType U;
  typedef typename U::value_type T;

  void single(char *dst, char *const *src)
  {
    PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);
    if (PyComplex_Check(src_obj)) {
      Py_complex v = PyComplex_AsCComplex(src_obj);
      if (v.real == -1 && PyErr_Occurred()) {
        throw std::exception();
      }
      reinterpret_cast<T *>(dst)[0] = static_cast<T>(v.real);
      reinterpret_cast<T *>(dst)[1] = static_cast<T>(v.imag);
    }
    else {
      *reinterpret_cast<dynd::complex<T> *>(dst) = pydynd::array_from_py(src_obj, 0, false).as<dynd::complex<T>>();
    }
  }
};

template <>
struct assign_from_pyobject_kernel<dynd::bytes> : nd::base_strided_kernel<assign_from_pyobject_kernel<dynd::bytes>, 1> {
  ndt::type dst_tp;
  const char *dst_arrmeta;

  assign_from_pyobject_kernel(const dynd::ndt::type &dst_tp, const char *dst_arrmeta)
      : dst_tp(dst_tp), dst_arrmeta(dst_arrmeta)
  {
  }

  void single(char *dst, char *const *src)
  {
    PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);
    char *pybytes_data = NULL;
    intptr_t pybytes_len = 0;
    if (PyBytes_Check(src_obj)) {
      if (PyBytes_AsStringAndSize(src_obj, &pybytes_data, &pybytes_len) < 0) {
        throw std::runtime_error("Error getting byte string data");
      }
    }
    else if (PyObject_TypeCheck(src_obj, pydynd::get_array_pytypeobject())) {
      pydynd::nd::typed_data_assign(dst_tp, dst_arrmeta, dst, pydynd::array_to_cpp_ref(src_obj));
      return;
    }
    else {
      std::stringstream ss;
      ss << "Cannot assign object " << pydynd::pyobject_repr(src_obj) << " to a dynd bytes value";
      throw std::invalid_argument(ss.str());
    }

    dynd::ndt::type bytes_tp = dynd::ndt::make_type<dynd::ndt::bytes_type>(1);
    dynd::string bytes_d(pybytes_data, pybytes_len);

    pydynd::nd::typed_data_assign(dst_tp, dst_arrmeta, dst, bytes_tp, NULL, reinterpret_cast<const char *>(&bytes_d));
  }
};

template <>
struct assign_from_pyobject_kernel<dynd::ndt::fixed_bytes_type> : assign_from_pyobject_kernel<dynd::bytes> {
};

template <>
struct assign_from_pyobject_kernel<dynd::string>
    : nd::base_strided_kernel<assign_from_pyobject_kernel<dynd::string>, 1> {
  ndt::type dst_tp;
  const char *dst_arrmeta;

  assign_from_pyobject_kernel(const ndt::type &dst_tp, const char *dst_arrmeta)
      : dst_tp(dst_tp), dst_arrmeta(dst_arrmeta)
  {
  }

  void single(char *dst, char *const *src)
  {
    PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);

    char *pybytes_data = NULL;
    intptr_t pybytes_len = 0;
    if (PyUnicode_Check(src_obj)) {
      // Go through UTF8 (was accessing the cpython unicode values directly
      // before, but on Python 3.3 OS X it didn't work correctly.)
      pydynd::py_ref utf8 = pydynd::capture_if_not_null(PyUnicode_AsUTF8String(src_obj));
      char *s = NULL;
      Py_ssize_t len = 0;
      if (PyBytes_AsStringAndSize(utf8.get(), &s, &len) < 0) {
        throw std::exception();
      }

      dynd::ndt::type str_tp = dynd::ndt::make_type<dynd::ndt::string_type>();
      dynd::string str_d(s, len);

      pydynd::nd::typed_data_assign(dst_tp, dst_arrmeta, dst, str_tp, NULL, reinterpret_cast<const char *>(&str_d));
#if PY_VERSION_HEX < 0x03000000
    }
    else if (PyString_Check(src_obj)) {
      char *pystr_data = NULL;
      intptr_t pystr_len = 0;

      if (PyString_AsStringAndSize(src_obj, &pystr_data, &pystr_len) < 0) {
        throw std::runtime_error("Error getting string data");
      }

      dynd::ndt::type str_dt = dynd::ndt::make_type<dynd::ndt::string_type>();
      dynd::string str_d(pystr_data, pystr_len);

      pydynd::nd::typed_data_assign(dst_tp, dst_arrmeta, dst, str_dt, NULL, reinterpret_cast<const char *>(&str_d));
#endif
    }
    else if (PyObject_TypeCheck(src_obj, pydynd::get_array_pytypeobject())) {
      pydynd::nd::typed_data_assign(dst_tp, dst_arrmeta, dst, pydynd::array_to_cpp_ref(src_obj));
      return;
    }
    else {
      std::stringstream ss;
      ss << "Cannot assign object " << pydynd::pyobject_repr(src_obj) << " to a dynd bytes value";
      throw std::invalid_argument(ss.str());
    }
  }
};

template <>
struct assign_from_pyobject_kernel<dynd::ndt::fixed_string_type> : assign_from_pyobject_kernel<dynd::string> {
};

template <>
struct assign_from_pyobject_kernel<dynd::ndt::type>
    : dynd::nd::base_strided_kernel<assign_from_pyobject_kernel<dynd::ndt::type>, 1> {
  void single(char *dst, char *const *src)
  {
    PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);
    *reinterpret_cast<dynd::ndt::type *>(dst) = pydynd::dynd_ndt_as_cpp_type(src_obj);
  }
};

template <>
struct assign_from_pyobject_kernel<dynd::ndt::option_type>
    : nd::base_strided_kernel<assign_from_pyobject_kernel<dynd::ndt::option_type>, 1> {
  dynd::ndt::type dst_tp;
  const char *dst_arrmeta;
  intptr_t copy_value_offset;

  assign_from_pyobject_kernel(const dynd::ndt::type &dst_tp, const char *dst_arrmeta)
      : dst_tp(dst_tp), dst_arrmeta(dst_arrmeta)
  {
  }

  ~assign_from_pyobject_kernel()
  {
    get_child()->destroy();
    get_child(copy_value_offset)->destroy();
  }

  void single(char *dst, char *const *src)
  {
    PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);
    if (src_obj == Py_None) {
      nd::kernel_prefix *assign_na = get_child();
      dynd::kernel_single_t assign_na_fn = assign_na->get_function<dynd::kernel_single_t>();
      assign_na_fn(assign_na, dst, NULL);
    }
    else if (PyObject_TypeCheck(src_obj, pydynd::get_array_pytypeobject())) {
      pydynd::nd::typed_data_assign(dst_tp, dst_arrmeta, dst, pydynd::array_to_cpp_ref(src_obj));
    }
    else if (dst_tp.get_base_id() != dynd::string_kind_id && PyUnicode_Check(src_obj)) {
      // Copy from the string
      pydynd::py_ref utf8 = pydynd::capture_if_not_null(PyUnicode_AsUTF8String(src_obj));
      char *s = NULL;
      Py_ssize_t len = 0;
      if (PyBytes_AsStringAndSize(utf8.get(), &s, &len) < 0) {
        throw std::exception();
      }

      dynd::ndt::type str_tp = dynd::ndt::make_type<dynd::ndt::string_type>();
      dynd::string str_d(s, len);
      const char *src_str = reinterpret_cast<const char *>(&str_d);

      pydynd::nd::typed_data_assign(dst_tp, dst_arrmeta, dst, str_tp, NULL, reinterpret_cast<const char *>(&str_d));
#if PY_VERSION_HEX < 0x03000000
    }
    else if (dst_tp.get_base_id() != dynd::string_kind_id && PyString_Check(src_obj)) {
      // Copy from the string
      char *s = NULL;
      Py_ssize_t len = 0;
      if (PyString_AsStringAndSize(src_obj, &s, &len) < 0) {
        throw std::exception();
      }

      dynd::ndt::type str_tp = dynd::ndt::make_type<dynd::ndt::string_type>();
      dynd::string str_d(s, len);
      const char *src_str = reinterpret_cast<const char *>(&str_d);

      pydynd::nd::typed_data_assign(dst_tp, dst_arrmeta, dst, str_tp, NULL, reinterpret_cast<const char *>(&str_d));
#endif
    }
    else {
      nd::kernel_prefix *copy_value = get_child(copy_value_offset);
      dynd::kernel_single_t copy_value_fn = copy_value->get_function<dynd::kernel_single_t>();
      copy_value_fn(copy_value, dst, src);
    }
  }
};

// TODO: Should make a more efficient strided kernel function
template <>
struct assign_from_pyobject_kernel<dynd::ndt::tuple_type>
    : dynd::nd::base_strided_kernel<assign_from_pyobject_kernel<dynd::ndt::tuple_type>, 1> {
  dynd::ndt::type m_dst_tp;
  const char *m_dst_arrmeta;
  bool m_dim_broadcast;
  std::vector<intptr_t> m_copy_el_offsets;

  ~assign_from_pyobject_kernel()
  {
    for (size_t i = 0; i < m_copy_el_offsets.size(); ++i) {
      get_child(m_copy_el_offsets[i])->destroy();
    }
  }

  void single(char *dst, char *const *src)
  {
    PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);

    if (PyObject_TypeCheck(src_obj, pydynd::get_array_pytypeobject())) {
      pydynd::nd::typed_data_assign(m_dst_tp, m_dst_arrmeta, dst, pydynd::array_to_cpp_ref(src_obj));
      return;
    }
#ifdef DYND_NUMPY_INTEROP
    if (PyArray_Check(src_obj)) {
      pydynd::nd::array_copy_from_numpy(m_dst_tp, m_dst_arrmeta, dst, (PyArrayObject *)src_obj,
                                        &dynd::eval::default_eval_context);
      return;
    }
#endif
    // TODO: PEP 3118 support here

    intptr_t field_count = m_dst_tp.extended<dynd::ndt::tuple_type>()->get_field_count();
    const uintptr_t *field_offsets = reinterpret_cast<const uintptr_t *>(m_dst_arrmeta);

    // Get the input as an array of PyObject *
    pydynd::py_ref src_fast;
    char *child_src;
    intptr_t child_stride = sizeof(PyObject *);
    intptr_t src_dim_size;
    if (m_dim_broadcast && pydynd::broadcast_as_scalar(m_dst_tp, src_obj)) {
      child_src = src[0];
      src_dim_size = 1;
    }
    else {
      src_fast = pydynd::capture_if_not_null(PySequence_Fast(src_obj, "Require a sequence to copy to a dynd tuple"));
      child_src = reinterpret_cast<char *>(PySequence_Fast_ITEMS(src_fast.get()));
      src_dim_size = PySequence_Fast_GET_SIZE(src_fast.get());
    }

    if (src_dim_size != 1 && field_count != src_dim_size) {
      std::stringstream ss;
      ss << "Cannot assign python value " << pydynd::pyobject_repr(src_obj) << " to a dynd " << m_dst_tp << " value";
      throw dynd::broadcast_error(ss.str());
    }
    if (src_dim_size == 1) {
      child_stride = 0;
    }
    for (intptr_t i = 0; i < field_count; ++i) {
      nd::kernel_prefix *copy_el = get_child(m_copy_el_offsets[i]);
      dynd::kernel_single_t copy_el_fn = copy_el->get_function<dynd::kernel_single_t>();
      char *el_dst = dst + field_offsets[i];
      char *el_src = child_src + i * child_stride;
      copy_el_fn(copy_el, el_dst, &el_src);
    }
    if (PyErr_Occurred()) {
      throw std::exception();
    }
  }
};

// TODO: Should make a more efficient strided kernel function
template <>
struct assign_from_pyobject_kernel<dynd::ndt::struct_type>
    : dynd::nd::base_strided_kernel<assign_from_pyobject_kernel<dynd::ndt::struct_type>, 1> {
  dynd::ndt::type m_dst_tp;
  const char *m_dst_arrmeta;
  bool m_dim_broadcast;
  std::vector<intptr_t> m_copy_el_offsets;

  ~assign_from_pyobject_kernel()
  {
    for (size_t i = 0; i < m_copy_el_offsets.size(); ++i) {
      get_child(m_copy_el_offsets[i])->destroy();
    }
  }

  void single(char *dst, char *const *src)
  {
    PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);

    if (PyObject_TypeCheck(src_obj, pydynd::get_array_pytypeobject())) {
      pydynd::nd::typed_data_assign(m_dst_tp, m_dst_arrmeta, dst, pydynd::array_to_cpp_ref(src_obj));
      return;
    }
#ifdef DYND_NUMPY_INTEROP
    if (PyArray_Check(src_obj)) {
      pydynd::nd::array_copy_from_numpy(m_dst_tp, m_dst_arrmeta, dst, (PyArrayObject *)src_obj,
                                        &dynd::eval::default_eval_context);
      return;
    }
#endif
    // TODO: PEP 3118 support here

    intptr_t field_count = m_dst_tp.extended<dynd::ndt::tuple_type>()->get_field_count();
    const uintptr_t *field_offsets = reinterpret_cast<const uintptr_t *>(m_dst_arrmeta);

    if (PyDict_Check(src_obj)) {
      // Keep track of which fields we've seen
      dynd::shortvector<bool> populated_fields(field_count);
      memset(populated_fields.get(), 0, sizeof(bool) * field_count);

      PyObject *dict_key = NULL, *dict_value = NULL;
      Py_ssize_t dict_pos = 0;

      while (PyDict_Next(src_obj, &dict_pos, &dict_key, &dict_value)) {
        std::string name = pydynd::pystring_as_string(dict_key);
        intptr_t i = m_dst_tp.extended<dynd::ndt::struct_type>()->get_field_index(name);
        // TODO: Add an error policy of whether to throw an error
        //       or not. For now, just raise an error
        if (i >= 0) {
          nd::kernel_prefix *copy_el = get_child(m_copy_el_offsets[i]);
          dynd::kernel_single_t copy_el_fn = copy_el->get_function<dynd::kernel_single_t>();
          char *el_dst = dst + field_offsets[i];
          char *el_src = reinterpret_cast<char *>(&dict_value);
          copy_el_fn(copy_el, el_dst, &el_src);
          populated_fields[i] = true;
        }
        else {
          std::stringstream ss;
          ss << "Input python dict has key ";
          dynd::print_escaped_utf8_string(ss, name);
          ss << ", but no such field is in destination dynd type " << m_dst_tp;
          throw dynd::broadcast_error(ss.str());
        }
      }

      for (intptr_t i = 0; i < field_count; ++i) {
        if (!populated_fields[i]) {
          std::stringstream ss;
          ss << "python dict does not contain the field ";
          dynd::print_escaped_utf8_string(ss, m_dst_tp.extended<dynd::ndt::struct_type>()->get_field_name(i));
          ss << " as required by the data type " << m_dst_tp;
          throw dynd::broadcast_error(ss.str());
        }
      }
    }
    else {
      // Get the input as an array of PyObject *
      pydynd::py_ref src_fast;
      char *child_src;
      intptr_t child_stride = sizeof(PyObject *);
      intptr_t src_dim_size;
      if (m_dim_broadcast && pydynd::broadcast_as_scalar(m_dst_tp, src_obj)) {
        child_src = src[0];
        src_dim_size = 1;
      }
      else {
        src_fast = pydynd::capture_if_not_null(PySequence_Fast(src_obj, "Require a sequence to copy to a dynd struct"));
        child_src = reinterpret_cast<char *>(PySequence_Fast_ITEMS(src_fast.get()));
        src_dim_size = PySequence_Fast_GET_SIZE(src_fast.get());
      }

      if (src_dim_size != 1 && field_count != src_dim_size) {
        std::stringstream ss;
        ss << "Cannot assign python value " << pydynd::pyobject_repr(src_obj) << " to a dynd " << m_dst_tp << " value";
        throw dynd::broadcast_error(ss.str());
      }
      if (src_dim_size == 1) {
        child_stride = 0;
      }
      for (intptr_t i = 0; i < field_count; ++i) {
        nd::kernel_prefix *copy_el = get_child(m_copy_el_offsets[i]);
        dynd::kernel_single_t copy_el_fn = copy_el->get_function<dynd::kernel_single_t>();
        char *el_dst = dst + field_offsets[i];
        char *el_src = child_src + i * child_stride;
        copy_el_fn(copy_el, el_dst, &el_src);
      }
    }
    if (PyErr_Occurred()) {
      throw std::exception();
    }
  }
};

// TODO: Could create the dst_tp -> dst_tp assignment
//       as part of the ckernel instead of dynamically
template <>
struct assign_from_pyobject_kernel<dynd::ndt::fixed_dim_type>
    : dynd::nd::base_strided_kernel<assign_from_pyobject_kernel<dynd::ndt::fixed_dim_type>, 1> {
  intptr_t m_dim_size, m_stride;
  dynd::ndt::type m_dst_tp;
  const char *m_dst_arrmeta;
  // Offset to ckernel which copies from dst to dst, for broadcasting case
  intptr_t m_copy_dst_offset;

  ~assign_from_pyobject_kernel() { get_child()->destroy(); }

  void single(char *dst, char *const *src)
  {
    PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);

    if (PyObject_TypeCheck(src_obj, pydynd::get_array_pytypeobject())) {
      pydynd::nd::typed_data_assign(m_dst_tp, m_dst_arrmeta, dst, pydynd::array_to_cpp_ref(src_obj));
      return;
    }
#ifdef DYND_NUMPY_INTEROP
    if (PyArray_Check(src_obj)) {
      pydynd::nd::array_copy_from_numpy(m_dst_tp, m_dst_arrmeta, dst, (PyArrayObject *)src_obj,
                                        &dynd::eval::default_eval_context);
      return;
    }
#endif
    // TODO: PEP 3118 support here

    nd::kernel_prefix *copy_el = get_child();
    dynd::kernel_strided_t copy_el_fn = copy_el->get_function<dynd::kernel_strided_t>();

    // Get the input as an array of PyObject *
    pydynd::py_ref src_fast;
    char *child_src;
    intptr_t child_stride = sizeof(PyObject *);
    intptr_t src_dim_size;

    if (!PyList_Check(src_obj) && pydynd::broadcast_as_scalar(m_dst_tp, src_obj)) {
      child_src = src[0];
      src_dim_size = 1;
    }
    else {
      src_fast =
          pydynd::capture_if_not_null(PySequence_Fast(src_obj, "Require a sequence to copy to a dynd dimension"));
      child_src = reinterpret_cast<char *>(PySequence_Fast_ITEMS(src_fast.get()));
      src_dim_size = PySequence_Fast_GET_SIZE(src_fast.get());
    }

    if (src_dim_size != 1 && m_dim_size != src_dim_size) {
      std::stringstream ss;
      ss << "Cannot assign python value " << pydynd::pyobject_repr(src_obj) << " to a dynd " << m_dst_tp << " value";
      throw dynd::broadcast_error(ss.str());
    }
    if (src_dim_size == 1 && m_dim_size > 1) {
      // Copy once from Python, then duplicate that element
      copy_el_fn(copy_el, dst, 0, &child_src, &child_stride, 1);
      nd::kernel_prefix *copy_dst = get_child(m_copy_dst_offset);
      dynd::kernel_strided_t copy_dst_fn = copy_dst->get_function<dynd::kernel_strided_t>();
      intptr_t zero = 0;
      copy_dst_fn(copy_dst, dst + m_stride, m_stride, &dst, &zero, m_dim_size - 1);
    }
    else {
      copy_el_fn(copy_el, dst, m_stride, &child_src, &child_stride, m_dim_size);
    }
    if (PyErr_Occurred()) {
      throw std::exception();
    }
  }
};

template <>
struct assign_from_pyobject_kernel<dynd::ndt::var_dim_type>
    : dynd::nd::base_strided_kernel<assign_from_pyobject_kernel<dynd::ndt::var_dim_type>, 1> {
  intptr_t m_offset, m_stride;
  dynd::ndt::type m_dst_tp;
  const char *m_dst_arrmeta;
  // Offset to ckernel which copies from dst to dst, for broadcasting case
  intptr_t m_copy_dst_offset;

  ~assign_from_pyobject_kernel() { get_child()->destroy(); }

  void single(char *dst, char *const *src)
  {
    PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);

    if (PyObject_TypeCheck(src_obj, pydynd::get_array_pytypeobject())) {
      pydynd::nd::typed_data_assign(m_dst_tp, m_dst_arrmeta, dst, pydynd::array_to_cpp_ref(src_obj));
      return;
    }
#ifdef DYND_NUMPY_INTEROP
    if (PyArray_Check(src_obj)) {
      pydynd::nd::array_copy_from_numpy(m_dst_tp, m_dst_arrmeta, dst, (PyArrayObject *)src_obj,
                                        &dynd::eval::default_eval_context);
      return;
    }
#endif
    // TODO: PEP 3118 support here

    nd::kernel_prefix *copy_el = get_child();
    dynd::kernel_strided_t copy_el_fn = copy_el->get_function<dynd::kernel_strided_t>();

    // Get the input as an array of PyObject *
    pydynd::py_ref src_fast;
    char *child_src;
    intptr_t child_stride = sizeof(PyObject *);
    intptr_t src_dim_size;
    if (!PyList_Check(src_obj) && pydynd::broadcast_as_scalar(m_dst_tp, src_obj)) {
      child_src = src[0];
      src_dim_size = 1;
    }
    else {
      src_fast =
          pydynd::capture_if_not_null(PySequence_Fast(src_obj, "Require a sequence to copy to a dynd dimension"));
      child_src = reinterpret_cast<char *>(PySequence_Fast_ITEMS(src_fast.get()));
      src_dim_size = PySequence_Fast_GET_SIZE(src_fast.get());
    }

    // If the var dim element hasn't been allocated, initialize it
    dynd::ndt::var_dim_type::data_type *vdd = reinterpret_cast<dynd::ndt::var_dim_type::data_type *>(dst);
    if (vdd->begin == NULL) {
      if (m_offset != 0) {
        throw std::runtime_error("Cannot assign to an uninitialized dynd var_dim "
                                 "which has a non-zero offset");
      }

      vdd->begin =
          reinterpret_cast<const ndt::var_dim_type::metadata_type *>(m_dst_arrmeta)->blockref->alloc(src_dim_size);
      vdd->size = src_dim_size;
    }

    if (src_dim_size != 1 && vdd->size != src_dim_size) {
      std::stringstream ss;
      ss << "Cannot assign python value " << pydynd::pyobject_repr(src_obj) << " to a dynd " << m_dst_tp << " value";
      throw dynd::broadcast_error(ss.str());
    }
    if (src_dim_size == 1 && vdd->size > 1) {
      // Copy once from Python, then duplicate that element
      copy_el_fn(copy_el, vdd->begin + m_offset, 0, &child_src, &child_stride, 1);
      nd::kernel_prefix *copy_dst = get_child(m_copy_dst_offset);
      dynd::kernel_strided_t copy_dst_fn = copy_dst->get_function<dynd::kernel_strided_t>();
      intptr_t zero = 0;
      char *src_to_dup = vdd->begin + m_offset;
      copy_dst_fn(copy_dst, vdd->begin + m_offset + m_stride, m_stride, &src_to_dup, &zero, vdd->size - 1);
    }
    else {
      copy_el_fn(copy_el, vdd->begin + m_offset, m_stride, &child_src, &child_stride, vdd->size);
    }
    if (PyErr_Occurred()) {
      throw std::exception();
    }
  }
};

} // unnamed namespace

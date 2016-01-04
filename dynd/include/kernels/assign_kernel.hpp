#pragma once

#include <dynd/kernels/base_kernel.hpp>

#include <Python.h>

#include "types/pyobject_type.hpp"

using namespace dynd;

namespace detail {

template <type_id_t DstTypeID, type_id_t DstBaseTypeID>
struct assign_kernel;

template <>
struct assign_kernel<bool_type_id, bool_kind_type_id>
    : nd::base_kernel<assign_kernel<bool_type_id, bool_kind_type_id>, 1> {
  void single(char *dst, char *const *src)
  {
    PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);
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
  pydynd::pyobject_ownref sixtyfour(PyLong_FromLong(64));
  pydynd::pyobject_ownref value_shr1(PyNumber_Rshift(obj, sixtyfour.get()));
  uint64_t hi = PyLong_AsUnsignedLongLongMask(value_shr1.get());
  dynd::int128 result(hi, lo);

  // Shift right another 64 bits, and check that nothing is remaining
  pydynd::pyobject_ownref value_shr2(
      PyNumber_Rshift(value_shr1.get(), sixtyfour.get()));
  long remaining = PyLong_AsLong(value_shr2.get());
  if ((remaining != 0 || (remaining == 0 && result.is_negative())) &&
      (remaining != -1 || PyErr_Occurred() ||
       (remaining == -1 && !result.is_negative()))) {
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
  pydynd::pyobject_ownref sixtyfour(PyLong_FromLong(64));
  pydynd::pyobject_ownref value_shr1(PyNumber_Rshift(obj, sixtyfour.get()));
  uint64_t hi = PyLong_AsUnsignedLongLongMask(value_shr1.get());
  dynd::uint128 result(hi, lo);

  // Shift right another 64 bits, and check that nothing is remaining
  pydynd::pyobject_ownref value_shr2(
      PyNumber_Rshift(value_shr1.get(), sixtyfour.get()));
  long remaining = PyLong_AsLong(value_shr2.get());
  if (remaining != 0) {
    throw std::overflow_error("int is too big to fit in an uint128");
  }

  *out = result;
}

template <type_id_t DstTypeID>
struct assign_kernel<DstTypeID, int_kind_type_id>
    : dynd::nd::base_kernel<assign_kernel<DstTypeID, int_kind_type_id>, 1> {
  typedef typename type_of<DstTypeID>::type T;

  void single(char *dst, char *const *src)
  {
    PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);
    if (PyLong_Check(src_obj)
#if PY_VERSION_HEX < 0x03000000
        || PyInt_Check(src_obj)
#endif
            ) {
      pyint_to_int(reinterpret_cast<T *>(dst), src_obj);
    }
    else {
      *reinterpret_cast<T *>(dst) =
          pydynd::array_from_py(src_obj, 0, false).as<T>();
    }
  }
};

template <type_id_t DstTypeID>
struct assign_kernel<DstTypeID, uint_kind_type_id>
    : dynd::nd::base_kernel<assign_kernel<DstTypeID, uint_kind_type_id>, 1> {
  typedef typename type_of<DstTypeID>::type T;

  void single(char *dst, char *const *src)
  {
    PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);
    if (PyLong_Check(src_obj)
#if PY_VERSION_HEX < 0x03000000
        || PyInt_Check(src_obj)
#endif
            ) {
      pyint_to_int(reinterpret_cast<T *>(dst), src_obj);
    }
    else {
      *reinterpret_cast<T *>(dst) =
          pydynd::array_from_py(src_obj, 0, false).as<T>();
    }
  }
};

template <>
struct assign_kernel<option_type_id, any_kind_type_id>
    : nd::base_kernel<assign_kernel<option_type_id, any_kind_type_id>, 1> {
  dynd::ndt::type dst_tp;
  const char *dst_arrmeta;
  intptr_t copy_value_offset;

  assign_kernel(const dynd::ndt::type &dst_tp, const char *dst_arrmeta)
      : dst_tp(dst_tp), dst_arrmeta(dst_arrmeta)
  {
  }

  ~assign_kernel()
  {
    get_child()->destroy();
    get_child(copy_value_offset)->destroy();
  }

  void single(char *dst, char *const *src)
  {
    PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);
    if (src_obj == Py_None) {
      ckernel_prefix *assign_na = get_child();
      dynd::kernel_single_t assign_na_fn =
          assign_na->get_function<dynd::kernel_single_t>();
      assign_na_fn(assign_na, dst, NULL);
    }
    else if (DyND_PyArray_Check(src_obj)) {
      pydynd::nd::typed_data_assign(dst_tp, dst_arrmeta, dst,
                                    ((DyND_PyArrayObject *)src_obj)->v);
    }
    else if (dst_tp.get_kind() != dynd::string_kind &&
             PyUnicode_Check(src_obj)) {
      // Copy from the string
      pydynd::pyobject_ownref utf8(PyUnicode_AsUTF8String(src_obj));
      char *s = NULL;
      Py_ssize_t len = 0;
      if (PyBytes_AsStringAndSize(utf8.get(), &s, &len) < 0) {
        throw std::exception();
      }

      dynd::ndt::type str_tp = dynd::ndt::make_type<dynd::ndt::string_type>();
      dynd::string str_d(s, len);
      const char *src_str = reinterpret_cast<const char *>(&str_d);

      pydynd::nd::typed_data_assign(dst_tp, dst_arrmeta, dst, str_tp, NULL,
                                    reinterpret_cast<const char *>(&str_d));
#if PY_VERSION_HEX < 0x03000000
    }
    else if (dst_tp.get_kind() != dynd::string_kind &&
             PyString_Check(src_obj)) {
      // Copy from the string
      char *s = NULL;
      Py_ssize_t len = 0;
      if (PyString_AsStringAndSize(src_obj, &s, &len) < 0) {
        throw std::exception();
      }

      dynd::ndt::type str_tp = dynd::ndt::make_type<dynd::ndt::string_type>();
      dynd::string str_d(s, len);
      const char *src_str = reinterpret_cast<const char *>(&str_d);

      pydynd::nd::typed_data_assign(dst_tp, dst_arrmeta, dst, str_tp, NULL,
                                    reinterpret_cast<const char *>(&str_d));
#endif
    }
    else {
      ckernel_prefix *copy_value = get_child(copy_value_offset);
      dynd::kernel_single_t copy_value_fn =
          copy_value->get_function<dynd::kernel_single_t>();
      copy_value_fn(copy_value, dst, src);
    }
  }

  static intptr_t
  instantiate(char *static_data, char *data, void *ckb, intptr_t ckb_offset,
              const dynd::ndt::type &dst_tp, const char *dst_arrmeta,
              intptr_t nsrc, const dynd::ndt::type *src_tp,
              const char *const *src_arrmeta, dynd::kernel_request_t kernreq,
              intptr_t nkwd, const dynd::nd::array *kwds,
              const std::map<std::string, dynd::ndt::type> &tp_vars)
  {

    intptr_t root_ckb_offset = ckb_offset;
    make(ckb, kernreq, ckb_offset, dst_tp, dst_arrmeta);
    dynd::nd::callable assign_na = dynd::nd::assign_na::get();
    ckb_offset = assign_na.get()->instantiate(
        assign_na.get()->static_data(), NULL, ckb, ckb_offset, dst_tp,
        dst_arrmeta, nsrc, NULL, NULL, dynd::kernel_request_single, nkwd, kwds,
        tp_vars);
    assign_kernel *self = get_self(
        reinterpret_cast<dynd::ckernel_builder<dynd::kernel_request_host> *>(
            ckb),
        root_ckb_offset);
    self->copy_value_offset = ckb_offset - root_ckb_offset;
    ckb_offset = nd::assign::get()->instantiate(
        nd::assign::get()->static_data(), NULL, ckb, ckb_offset,
        dst_tp.extended<dynd::ndt::option_type>()->get_value_type(),
        dst_arrmeta, nsrc, src_tp, src_arrmeta, dynd::kernel_request_single,
        nkwd, kwds, tp_vars);
    return ckb_offset;
  }
};

} // namespace detail

template <type_id_t DstTypeID>
using assign_kernel =
    ::detail::assign_kernel<DstTypeID, base_type_id_of<DstTypeID>::value>;

namespace dynd {
namespace ndt {

  template <type_id_t DstTypeID>
  struct traits<assign_kernel<DstTypeID>> {
    static type equivalent()
    {
      return callable_type::make(DstTypeID, {ndt::make_type<pyobject_type>()});
    }
  };

} // namespace dynd::ndt
} // namespace dynd

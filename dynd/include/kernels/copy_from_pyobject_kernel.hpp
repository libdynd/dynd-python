#pragma once

#include <dynd/parser_util.hpp>
#include <dynd/func/compose.hpp>
#include <dynd/func/copy.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/base_virtual_kernel.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/categorical_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/datetime_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/time_type.hpp>
#include <dynd/types/var_dim_type.hpp>

#include "config.hpp"

namespace pydynd {
namespace nd {

  template <dynd::type_id_t src_type_id>
  struct copy_from_pyobject_kernel;

  template <>
  struct copy_from_pyobject_kernel<
      dynd::bool_type_id> : dynd::nd::
                                base_kernel<copy_from_pyobject_kernel<
                                                dynd::bool_type_id>,
                                            1> {
    void single(char *dst, char *const *src)
    {
      PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);
      if (src_obj == Py_True) {
        *dst = 1;
      } else if (src_obj == Py_False) {
        *dst = 0;
      } else {
        *dst =
            array_from_py(src_obj, 0, false, &dynd::eval::default_eval_context)
                .as<dynd::bool1>();
      }
    }
  };

  void pyint_to_int(int8_t *out, PyObject *obj)
  {
    long v = PyLong_AsLong(obj);
    if (v == -1 && PyErr_Occurred()) {
      throw std::exception();
    }
    if (dynd::parse::overflow_check<int8_t>::is_overflow(v)) {
      throw std::overflow_error("overflow assigning to dynd int8");
    }
    *out = static_cast<int8_t>(v);
  }

  void pyint_to_int(int16_t *out, PyObject *obj)
  {
    long v = PyLong_AsLong(obj);
    if (v == -1 && PyErr_Occurred()) {
      throw std::exception();
    }
    if (dynd::parse::overflow_check<int16_t>::is_overflow(v)) {
      throw std::overflow_error("overflow assigning to dynd int16");
    }
    *out = static_cast<int16_t>(v);
  }

  void pyint_to_int(int32_t *out, PyObject *obj)
  {
    long v = PyLong_AsLong(obj);
    if (v == -1 && PyErr_Occurred()) {
      throw std::exception();
    }
    if (dynd::parse::overflow_check<int32_t>::is_overflow(v)) {
      throw std::overflow_error("overflow assigning to dynd int32");
    }
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
    pyobject_ownref sixtyfour(PyLong_FromLong(64));
    pyobject_ownref value_shr1(PyNumber_Rshift(obj, sixtyfour.get()));
    uint64_t hi = PyLong_AsUnsignedLongLongMask(value_shr1.get());
    dynd::int128 result(hi, lo);

    // Shift right another 64 bits, and check that nothing is remaining
    pyobject_ownref value_shr2(
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
    if (dynd::parse::overflow_check<uint8_t>::is_overflow(v)) {
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
    if (dynd::parse::overflow_check<uint16_t>::is_overflow(v)) {
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
    if (dynd::parse::overflow_check<uint32_t>::is_overflow(v)) {
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
    pyobject_ownref sixtyfour(PyLong_FromLong(64));
    pyobject_ownref value_shr1(PyNumber_Rshift(obj, sixtyfour.get()));
    uint64_t hi = PyLong_AsUnsignedLongLongMask(value_shr1.get());
    dynd::uint128 result(hi, lo);

    // Shift right another 64 bits, and check that nothing is remaining
    pyobject_ownref value_shr2(
        PyNumber_Rshift(value_shr1.get(), sixtyfour.get()));
    long remaining = PyLong_AsLong(value_shr2.get());
    if (remaining != 0) {
      throw std::overflow_error("int is too big to fit in an uint128");
    }

    *out = result;
  }

  template <typename T>
  struct copy_int_from_pyobject_kernel
      : dynd::nd::base_kernel<copy_int_from_pyobject_kernel<T>, 1> {
    void single(char *dst, char *const *src)
    {
      PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);
      if (PyLong_Check(src_obj)
#if PY_VERSION_HEX < 0x03000000
          || PyInt_Check(src_obj)
#endif
          ) {
        pyint_to_int(reinterpret_cast<T *>(dst), src_obj);
      } else {
        *reinterpret_cast<T *>(dst) = array_from_py(
            src_obj, 0, false, &dynd::eval::default_eval_context).as<T>();
      }
    }
  };

  template <>
  struct copy_from_pyobject_kernel<
      dynd::int8_type_id> : copy_int_from_pyobject_kernel<int8_t> {
  };

  template <>
  struct copy_from_pyobject_kernel<
      dynd::int16_type_id> : copy_int_from_pyobject_kernel<int16_t> {
  };

  template <>
  struct copy_from_pyobject_kernel<
      dynd::int32_type_id> : copy_int_from_pyobject_kernel<int32_t> {
  };

  template <>
  struct copy_from_pyobject_kernel<
      dynd::int64_type_id> : copy_int_from_pyobject_kernel<int64_t> {
  };

  template <>
  struct copy_from_pyobject_kernel<
      dynd::int128_type_id> : copy_int_from_pyobject_kernel<dynd::int128> {
  };

  template <>
  struct copy_from_pyobject_kernel<
      dynd::uint8_type_id> : copy_int_from_pyobject_kernel<uint8_t> {
  };

  template <>
  struct copy_from_pyobject_kernel<
      dynd::uint16_type_id> : copy_int_from_pyobject_kernel<uint16_t> {
  };

  template <>
  struct copy_from_pyobject_kernel<
      dynd::uint32_type_id> : copy_int_from_pyobject_kernel<uint32_t> {
  };

  template <>
  struct copy_from_pyobject_kernel<
      dynd::uint64_type_id> : copy_int_from_pyobject_kernel<uint64_t> {
  };

  template <>
  struct copy_from_pyobject_kernel<
      dynd::uint128_type_id> : copy_int_from_pyobject_kernel<dynd::uint128> {
  };

  template <typename T>
  struct float_copy_from_pyobject_kernel
      : dynd::nd::base_kernel<float_copy_from_pyobject_kernel<T>, 1> {
    void single(char *dst, char *const *src)
    {
      PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);
      if (PyFloat_Check(src_obj)) {
        double v = PyFloat_AsDouble(src_obj);
        if (v == -1 && PyErr_Occurred()) {
          throw std::exception();
        }
        *reinterpret_cast<T *>(dst) = static_cast<T>(v);
      } else {
        *reinterpret_cast<T *>(dst) = array_from_py(
            src_obj, 0, false, &dynd::eval::default_eval_context).as<T>();
      }
    }
  };

  template <>
  struct copy_from_pyobject_kernel<
      dynd::float16_type_id> : float_copy_from_pyobject_kernel<dynd::float16> {
  };

  template <>
  struct copy_from_pyobject_kernel<
      dynd::float32_type_id> : float_copy_from_pyobject_kernel<float> {
  };

  template <>
  struct copy_from_pyobject_kernel<
      dynd::float64_type_id> : float_copy_from_pyobject_kernel<double> {
  };

  template <typename T>
  struct complex_float_copy_from_pyobject_kernel
      : dynd::nd::base_kernel<complex_float_copy_from_pyobject_kernel<T>, 1> {
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
      } else {
        *reinterpret_cast<dynd::complex<T> *>(dst) =
            array_from_py(src_obj, 0, false, &dynd::eval::default_eval_context)
                .as<dynd::complex<T>>();
      }
    }
  };

  template <>
  struct copy_from_pyobject_kernel<
      dynd::
          complex_float32_type_id> : complex_float_copy_from_pyobject_kernel<float> {
  };

  template <>
  struct copy_from_pyobject_kernel<
      dynd::
          complex_float64_type_id> : complex_float_copy_from_pyobject_kernel<double> {
  };

  // TODO: This is not very efficient, could be made better
  //       with specialized bytes/fixed_bytes versions
  template <>
  struct copy_from_pyobject_kernel<
      dynd::bytes_type_id> : dynd::nd::
                                 base_kernel<copy_from_pyobject_kernel<
                                                 dynd::bytes_type_id>,
                                             1> {
    dynd::ndt::type dst_tp;
    const char *dst_arrmeta;

    copy_from_pyobject_kernel(const dynd::ndt::type &dst_tp,
                              const char *dst_arrmeta)
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
      } else if (DyND_PyArray_Check(src_obj)) {
        dynd::typed_data_assign(dst_tp, dst_arrmeta, dst,
                                ((DyND_PyArrayObject *)src_obj)->v);
        return;
      } else {
        std::stringstream ss;
        ss << "Cannot assign object " << pyobject_repr(src_obj)
           << " to a dynd bytes value";
        throw std::invalid_argument(ss.str());
      }

      dynd::ndt::type bytes_tp = dynd::ndt::bytes_type::make(1);
      dynd::string bytes_d(pybytes_data, pybytes_len);

      dynd::typed_data_assign(dst_tp, dst_arrmeta, dst, bytes_tp, NULL,
                              reinterpret_cast<const char *>(&bytes_d));
    }

    static intptr_t
    instantiate(char *static_data, size_t data_size, char *data, void *ckb,
                intptr_t ckb_offset, const dynd::ndt::type &dst_tp,
                const char *dst_arrmeta, intptr_t nsrc,
                const dynd::ndt::type *src_tp, const char *const *src_arrmeta,
                dynd::kernel_request_t kernreq,
                const dynd::eval::eval_context *ectx, intptr_t nkwd,
                const dynd::nd::array *kwds,
                const std::map<std::string, dynd::ndt::type> &tp_vars)
    {
      make(ckb, kernreq, ckb_offset, dst_tp, dst_arrmeta);
      return ckb_offset;
    }
  };

  template <>
  struct copy_from_pyobject_kernel<
      dynd::
          fixed_bytes_type_id> : copy_from_pyobject_kernel<dynd::
                                                               bytes_type_id> {
  };

  // TODO: This is not very efficient, could be made better
  //       with specialized versions
  template <>
  struct copy_from_pyobject_kernel<
      dynd::string_type_id> : dynd::nd::
                                  base_kernel<copy_from_pyobject_kernel<
                                                  dynd::string_type_id>,
                                              1> {
    dynd::ndt::type dst_tp;
    const char *dst_arrmeta;

    copy_from_pyobject_kernel(const dynd::ndt::type &dst_tp,
                              const char *dst_arrmeta)
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
        pyobject_ownref utf8(PyUnicode_AsUTF8String(src_obj));
        char *s = NULL;
        Py_ssize_t len = 0;
        if (PyBytes_AsStringAndSize(utf8.get(), &s, &len) < 0) {
          throw std::exception();
        }

        dynd::ndt::type str_tp = dynd::ndt::string_type::make();
        dynd::string str_d(s, len);

        dynd::typed_data_assign(dst_tp, dst_arrmeta, dst, str_tp, NULL,
                                reinterpret_cast<const char *>(&str_d));
#if PY_VERSION_HEX < 0x03000000
      } else if (PyString_Check(src_obj)) {
        char *pystr_data = NULL;
        intptr_t pystr_len = 0;

        if (PyString_AsStringAndSize(src_obj, &pystr_data, &pystr_len) < 0) {
          throw std::runtime_error("Error getting string data");
        }

        dynd::ndt::type str_dt = dynd::ndt::string_type::make();
        dynd::string str_d(pystr_data, pystr_len);

        dynd::typed_data_assign(dst_tp, dst_arrmeta, dst, str_dt, NULL,
                                reinterpret_cast<const char *>(&str_d));
#endif
      } else if (DyND_PyArray_Check(src_obj)) {
        dynd::typed_data_assign(dst_tp, dst_arrmeta, dst,
                                ((DyND_PyArrayObject *)src_obj)->v);
        return;
      } else {
        std::stringstream ss;
        ss << "Cannot assign object " << pyobject_repr(src_obj)
           << " to a dynd bytes value";
        throw std::invalid_argument(ss.str());
      }
    }

    static intptr_t
    instantiate(char *static_data, size_t data_size, char *data, void *ckb,
                intptr_t ckb_offset, const dynd::ndt::type &dst_tp,
                const char *dst_arrmeta, intptr_t nsrc,
                const dynd::ndt::type *src_tp, const char *const *src_arrmeta,
                dynd::kernel_request_t kernreq,
                const dynd::eval::eval_context *ectx, intptr_t nkwd,
                const dynd::nd::array *kwds,
                const std::map<std::string, dynd::ndt::type> &tp_vars)
    {
      make(ckb, kernreq, ckb_offset, dst_tp, dst_arrmeta);
      return ckb_offset;
    }
  };

  template <>
  struct copy_from_pyobject_kernel<
      dynd::
          fixed_string_type_id> : copy_from_pyobject_kernel<dynd::
                                                                string_type_id> {
  };

  template <>
  struct copy_from_pyobject_kernel<
      dynd::date_type_id> : dynd::nd::
                                base_kernel<copy_from_pyobject_kernel<
                                                dynd::date_type_id>,
                                            1> {
    dynd::ndt::type dst_tp;
    const char *dst_arrmeta;

    copy_from_pyobject_kernel(const dynd::ndt::type &dst_tp,
                              const char *dst_arrmeta)
        : dst_tp(dst_tp), dst_arrmeta(dst_arrmeta)
    {
    }

    void single(char *dst, char *const *src)
    {
      PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);
      if (PyDate_Check(src_obj)) {
        const dynd::ndt::date_type *dd =
            dst_tp.extended<dynd::ndt::date_type>();
        dd->set_ymd(dst_arrmeta, dst, dynd::assign_error_fractional,
                    PyDateTime_GET_YEAR(src_obj), PyDateTime_GET_MONTH(src_obj),
                    PyDateTime_GET_DAY(src_obj));
      } else if (PyDateTime_Check(src_obj)) {
        PyDateTime_DateTime *src_dt = (PyDateTime_DateTime *)src_obj;
        if (src_dt->hastzinfo && src_dt->tzinfo != NULL) {
          throw std::runtime_error(
              "Converting datetimes with a timezone to dynd "
              "arrays is not yet supported");
        }
        if (PyDateTime_DATE_GET_HOUR(src_obj) != 0 ||
            PyDateTime_DATE_GET_MINUTE(src_obj) != 0 ||
            PyDateTime_DATE_GET_SECOND(src_obj) != 0 ||
            PyDateTime_DATE_GET_MICROSECOND(src_obj) != 0) {
          std::stringstream ss;
          ss << "Cannot convert a datetime with non-zero time "
             << pyobject_repr(src_obj) << " to a datetime date";
          throw std::invalid_argument(ss.str());
        }
        const dynd::ndt::date_type *dd =
            dst_tp.extended<dynd::ndt::date_type>();
        dd->set_ymd(dst_arrmeta, dst, dynd::assign_error_fractional,
                    PyDateTime_GET_YEAR(src_obj), PyDateTime_GET_MONTH(src_obj),
                    PyDateTime_GET_DAY(src_obj));
      } else if (DyND_PyArray_Check(src_obj)) {
        dynd::typed_data_assign(dst_tp, dst_arrmeta, dst,
                                ((DyND_PyArrayObject *)src_obj)->v);
      } else {
        dynd::typed_data_assign(
            dst_tp, dst_arrmeta, dst,
            array_from_py(src_obj, 0, false,
                          &dynd::eval::default_eval_context));
      }
    }

    static intptr_t
    instantiate(char *static_data, size_t data_size, char *data, void *ckb,
                intptr_t ckb_offset, const dynd::ndt::type &dst_tp,
                const char *dst_arrmeta, intptr_t nsrc,
                const dynd::ndt::type *src_tp, const char *const *src_arrmeta,
                dynd::kernel_request_t kernreq,
                const dynd::eval::eval_context *ectx, intptr_t nkwd,
                const dynd::nd::array *kwds,
                const std::map<std::string, dynd::ndt::type> &tp_vars)
    {
      make(ckb, kernreq, ckb_offset, dst_tp, dst_arrmeta);
      return ckb_offset;
    }
  };

  template <>
  struct copy_from_pyobject_kernel<
      dynd::time_type_id> : dynd::nd::
                                base_kernel<copy_from_pyobject_kernel<
                                                dynd::time_type_id>,
                                            1> {

    dynd::ndt::type dst_tp;
    const char *dst_arrmeta;

    copy_from_pyobject_kernel(const dynd::ndt::type &dst_tp,
                              const char *dst_arrmeta)
        : dst_tp(dst_tp), dst_arrmeta(dst_arrmeta)
    {
    }

    void single(char *dst, char *const *src)
    {
      PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);
      if (PyTime_Check(src_obj)) {
        const dynd::ndt::time_type *tt =
            dst_tp.extended<dynd::ndt::time_type>();
        tt->set_time(dst_arrmeta, dst, dynd::assign_error_fractional,
                     PyDateTime_TIME_GET_HOUR(src_obj),
                     PyDateTime_TIME_GET_MINUTE(src_obj),
                     PyDateTime_TIME_GET_SECOND(src_obj),
                     PyDateTime_TIME_GET_MICROSECOND(src_obj) *
                         DYND_TICKS_PER_MICROSECOND);
      } else if (DyND_PyArray_Check(src_obj)) {
        dynd::typed_data_assign(dst_tp, dst_arrmeta, dst,
                                ((DyND_PyArrayObject *)src_obj)->v);
      } else {
        dynd::typed_data_assign(
            dst_tp, dst_arrmeta, dst,
            array_from_py(src_obj, 0, false,
                          &dynd::eval::default_eval_context));
      }
    }

    static intptr_t
    instantiate(char *static_data, size_t data_size, char *data, void *ckb,
                intptr_t ckb_offset, const dynd::ndt::type &dst_tp,
                const char *dst_arrmeta, intptr_t nsrc,
                const dynd::ndt::type *src_tp, const char *const *src_arrmeta,
                dynd::kernel_request_t kernreq,
                const dynd::eval::eval_context *ectx, intptr_t nkwd,
                const dynd::nd::array *kwds,
                const std::map<std::string, dynd::ndt::type> &tp_vars)
    {
      make(ckb, kernreq, ckb_offset, dst_tp, dst_arrmeta);
      return ckb_offset;
    }
  };

  template <>
  struct copy_from_pyobject_kernel<
      dynd::datetime_type_id> : dynd::nd::
                                    base_kernel<copy_from_pyobject_kernel<
                                                    dynd::datetime_type_id>,
                                                1> {
    dynd::ndt::type dst_tp;
    const char *dst_arrmeta;

    copy_from_pyobject_kernel(const dynd::ndt::type &dst_tp,
                              const char *dst_arrmeta)
        : dst_tp(dst_tp), dst_arrmeta(dst_arrmeta)
    {
    }

    void single(char *dst, char *const *src)
    {
      PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);
      if (PyDateTime_Check(src_obj)) {
        PyDateTime_DateTime *src_dt = (PyDateTime_DateTime *)src_obj;
        if (src_dt->hastzinfo && src_dt->tzinfo != NULL) {
          throw std::runtime_error(
              "Converting datetimes with a timezone to dynd "
              "arrays is not yet supported");
        }
        const dynd::ndt::datetime_type *dd =
            dst_tp.extended<dynd::ndt::datetime_type>();
        dd->set_cal(dst_arrmeta, dst, dynd::assign_error_fractional,
                    PyDateTime_GET_YEAR(src_obj), PyDateTime_GET_MONTH(src_obj),
                    PyDateTime_GET_DAY(src_obj),
                    PyDateTime_DATE_GET_HOUR(src_obj),
                    PyDateTime_DATE_GET_MINUTE(src_obj),
                    PyDateTime_DATE_GET_SECOND(src_obj),
                    PyDateTime_DATE_GET_MICROSECOND(src_obj) * 10);
      } else if (DyND_PyArray_Check(src_obj)) {
        dynd::typed_data_assign(dst_tp, dst_arrmeta, dst,
                                ((DyND_PyArrayObject *)src_obj)->v);
      } else {
        dynd::typed_data_assign(
            dst_tp, dst_arrmeta, dst,
            array_from_py(src_obj, 0, false,
                          &dynd::eval::default_eval_context));
      }
    }

    static intptr_t
    instantiate(char *static_data, size_t data_size, char *data, void *ckb,
                intptr_t ckb_offset, const dynd::ndt::type &dst_tp,
                const char *dst_arrmeta, intptr_t nsrc,
                const dynd::ndt::type *src_tp, const char *const *src_arrmeta,
                dynd::kernel_request_t kernreq,
                const dynd::eval::eval_context *ectx, intptr_t nkwd,
                const dynd::nd::array *kwds,
                const std::map<std::string, dynd::ndt::type> &tp_vars)
    {
      make(ckb, kernreq, ckb_offset, dst_tp, dst_arrmeta);
      return ckb_offset;
    }
  };

  template <>
  struct copy_from_pyobject_kernel<
      dynd::type_type_id> : dynd::nd::
                                base_kernel<copy_from_pyobject_kernel<
                                                dynd::type_type_id>,
                                            1> {
    void single(char *dst, char *const *src)
    {
      PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);
      *reinterpret_cast<dynd::ndt::type *>(dst) =
          pydynd::make__type_from_pyobject(src_obj);
    }
  };

  // TODO: Should make a more efficient strided kernel function
  template <>
  struct copy_from_pyobject_kernel<
      dynd::option_type_id> : dynd::nd::
                                  base_kernel<copy_from_pyobject_kernel<
                                                  dynd::option_type_id>,
                                              1> {
    dynd::ndt::type dst_tp;
    const char *dst_arrmeta;
    intptr_t copy_value_offset;

    copy_from_pyobject_kernel(const dynd::ndt::type &dst_tp,
                              const char *dst_arrmeta)
        : dst_tp(dst_tp), dst_arrmeta(dst_arrmeta)
    {
    }

    ~copy_from_pyobject_kernel()
    {
      get_child()->destroy();
      get_child(copy_value_offset)->destroy();
    }

    void single(char *dst, char *const *src)
    {
      PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);
      if (src_obj == Py_None) {
        ckernel_prefix *assign_na = get_child();
        dynd::expr_single_t assign_na_fn =
            assign_na->get_function<dynd::expr_single_t>();
        assign_na_fn(assign_na, dst, NULL);
      } else if (DyND_PyArray_Check(src_obj)) {
        dynd::typed_data_assign(dst_tp, dst_arrmeta, dst,
                                ((DyND_PyArrayObject *)src_obj)->v);
      } else if (dst_tp.get_kind() != dynd::string_kind &&
                 PyUnicode_Check(src_obj)) {
        // Copy from the string
        pyobject_ownref utf8(PyUnicode_AsUTF8String(src_obj));
        char *s = NULL;
        Py_ssize_t len = 0;
        if (PyBytes_AsStringAndSize(utf8.get(), &s, &len) < 0) {
          throw std::exception();
        }

        dynd::ndt::type str_tp = dynd::ndt::string_type::make();
        dynd::string str_d(s, len);
        const char *src_str = reinterpret_cast<const char *>(&str_d);

        dynd::typed_data_assign(dst_tp, dst_arrmeta, dst, str_tp, NULL,
                                reinterpret_cast<const char *>(&str_d));
#if PY_VERSION_HEX < 0x03000000
      } else if (dst_tp.get_kind() != dynd::string_kind &&
                 PyString_Check(src_obj)) {
        // Copy from the string
        char *s = NULL;
        Py_ssize_t len = 0;
        if (PyString_AsStringAndSize(src_obj, &s, &len) < 0) {
          throw std::exception();
        }

        dynd::ndt::type str_tp = dynd::ndt::string_type::make();
        dynd::string str_d(s, len);
        const char *src_str = reinterpret_cast<const char *>(&str_d);

        dynd::typed_data_assign(dst_tp, dst_arrmeta, dst, str_tp, NULL,
                                reinterpret_cast<const char *>(&str_d));
#endif
      } else {
        ckernel_prefix *copy_value = get_child(copy_value_offset);
        dynd::expr_single_t copy_value_fn =
            copy_value->get_function<dynd::expr_single_t>();
        copy_value_fn(copy_value, dst, src);
      }
    }

    static intptr_t
    instantiate(char *static_data, size_t data_size, char *data, void *ckb,
                intptr_t ckb_offset, const dynd::ndt::type &dst_tp,
                const char *dst_arrmeta, intptr_t nsrc,
                const dynd::ndt::type *src_tp, const char *const *src_arrmeta,
                dynd::kernel_request_t kernreq,
                const dynd::eval::eval_context *ectx, intptr_t nkwd,
                const dynd::nd::array *kwds,
                const std::map<std::string, dynd::ndt::type> &tp_vars)
    {

      intptr_t root_ckb_offset = ckb_offset;
      make(ckb, kernreq, ckb_offset, dst_tp, dst_arrmeta);
      dynd::nd::callable assign_na =
          dst_tp.extended<dynd::ndt::option_type>()->get_assign_na();
      ckb_offset = assign_na.get()->instantiate(
          assign_na.get()->static_data, 0, NULL, ckb, ckb_offset, dst_tp,
          dst_arrmeta, nsrc, NULL, NULL, dynd::kernel_request_single, ectx,
          nkwd, kwds, tp_vars);
      copy_from_pyobject_kernel *self = get_self(
          reinterpret_cast<dynd::ckernel_builder<dynd::kernel_request_host> *>(
              ckb),
          root_ckb_offset);
      self->copy_value_offset = ckb_offset - root_ckb_offset;
      ckb_offset = copy_from_pyobject::get().get()->instantiate(
          copy_from_pyobject::get().get()->static_data, 0, NULL, ckb,
          ckb_offset,
          dst_tp.extended<dynd::ndt::option_type>()->get_value_type(),
          dst_arrmeta, nsrc, src_tp, src_arrmeta, dynd::kernel_request_single,
          ectx, nkwd, kwds, tp_vars);
      return ckb_offset;
    }
  };

  template <>
  struct copy_from_pyobject_kernel<
      dynd::categorical_type_id> : dynd::nd::
                                       base_virtual_kernel<
                                           copy_from_pyobject_kernel<
                                               dynd::categorical_type_id>> {
    static intptr_t
    instantiate(char *static_data, size_t data_size, char *data, void *ckb,
                intptr_t ckb_offset, const dynd::ndt::type &dst_tp,
                const char *dst_arrmeta, intptr_t nsrc,
                const dynd::ndt::type *src_tp, const char *const *src_arrmeta,
                dynd::kernel_request_t kernreq,
                const dynd::eval::eval_context *ectx, intptr_t nkwd,
                const dynd::nd::array *kwds,
                const std::map<std::string, dynd::ndt::type> &tp_vars)
    {
      // Assign via an intermediate category_type buffer
      const dynd::ndt::type &buf_tp =
          dst_tp.extended<dynd::ndt::categorical_type>()->get_category_type();
      dynd::nd::callable copy_af = make_callable_from_assignment(
          dst_tp, buf_tp, dynd::assign_error_default);
      dynd::nd::callable child =
          dynd::nd::functional::compose(copy_from_pyobject, copy_af, buf_tp);
      return child.get()->instantiate(child.get()->static_data, 0, NULL, ckb,
                                      ckb_offset, dst_tp, dst_arrmeta, nsrc,
                                      src_tp, src_arrmeta, kernreq, ectx, 0,
                                      NULL, tp_vars);
    }
  };

  // TODO: Could instantiate the dst_tp -> dst_tp assignment
  //       as part of the ckernel instead of dynamically
  template <>
  struct copy_from_pyobject_kernel<
      dynd::fixed_dim_type_id> : dynd::nd::
                                     base_kernel<copy_from_pyobject_kernel<
                                                     dynd::fixed_dim_type_id>,
                                                 1> {
    intptr_t m_dim_size, m_stride;
    dynd::ndt::type m_dst_tp;
    const char *m_dst_arrmeta;
    bool m_dim_broadcast;
    // Offset to ckernel which copies from dst to dst, for broadcasting case
    intptr_t m_copy_dst_offset;

    ~copy_from_pyobject_kernel()
    {
      get_child()->destroy();
    }

    inline void single(char *dst, char *const *src)
    {
      PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);

      if (DyND_PyArray_Check(src_obj)) {
        dynd::typed_data_assign(m_dst_tp, m_dst_arrmeta, dst,
                                ((DyND_PyArrayObject *)src_obj)->v);
        return;
      }
#ifdef DYND_NUMPY_INTEROP
      if (PyArray_Check(src_obj)) {
        array_copy_from_numpy(m_dst_tp, m_dst_arrmeta, dst,
                              (PyArrayObject *)src_obj,
                              &dynd::eval::default_eval_context);
        return;
      }
#endif
      // TODO: PEP 3118 support here

      ckernel_prefix *copy_el = get_child();
      dynd::expr_strided_t copy_el_fn =
          copy_el->get_function<dynd::expr_strided_t>();

      // Get the input as an array of PyObject *
      pyobject_ownref src_fast;
      char *child_src;
      intptr_t child_stride = sizeof(PyObject *);
      intptr_t src_dim_size;
      if (m_dim_broadcast && broadcast_as_scalar(m_dst_tp, src_obj)) {
        child_src = src[0];
        src_dim_size = 1;
      } else {
        src_fast.reset(PySequence_Fast(
            src_obj, "Require a sequence to copy to a dynd dimension"));
        child_src =
            reinterpret_cast<char *>(PySequence_Fast_ITEMS(src_fast.get()));
        src_dim_size = PySequence_Fast_GET_SIZE(src_fast.get());
      }

      if (src_dim_size != 1 && m_dim_size != src_dim_size) {
        std::stringstream ss;
        ss << "Cannot assign python value " << pyobject_repr(src_obj)
           << " to a dynd " << m_dst_tp << " value";
        throw dynd::broadcast_error(ss.str());
      }
      if (src_dim_size == 1 && m_dim_size > 1) {
        // Copy once from Python, then duplicate that element
        copy_el_fn(copy_el, dst, 0, &child_src, &child_stride, 1);
        ckernel_prefix *copy_dst = get_child(m_copy_dst_offset);
        dynd::expr_strided_t copy_dst_fn =
            copy_dst->get_function<dynd::expr_strided_t>();
        intptr_t zero = 0;
        copy_dst_fn(copy_dst, dst + m_stride, m_stride, &dst, &zero,
                    m_dim_size - 1);
      } else {
        copy_el_fn(copy_el, dst, m_stride, &child_src, &child_stride,
                   m_dim_size);
      }
      if (PyErr_Occurred()) {
        throw std::exception();
      }
    }

    static intptr_t
    instantiate(char *static_data, size_t data_size, char *data, void *ckb,
                intptr_t ckb_offset, const dynd::ndt::type &dst_tp,
                const char *dst_arrmeta, intptr_t nsrc,
                const dynd::ndt::type *src_tp, const char *const *src_arrmeta,
                dynd::kernel_request_t kernreq,
                const dynd::eval::eval_context *ectx, intptr_t nkwd,
                const dynd::nd::array *kwds,
                const std::map<std::string, dynd::ndt::type> &tp_vars)
    {
      bool dim_broadcast = kwds[0].as<bool>();

      intptr_t dim_size, stride;
      dynd::ndt::type el_tp;
      const char *el_arrmeta;
      if (dst_tp.get_as_strided(dst_arrmeta, &dim_size, &stride, &el_tp,
                                &el_arrmeta)) {
        intptr_t root_ckb_offset = ckb_offset;
        pydynd::nd::copy_from_pyobject_kernel<dynd::fixed_dim_type_id> *self =
            pydynd::nd::copy_from_pyobject_kernel<
                dynd::fixed_dim_type_id>::make(ckb, kernreq, ckb_offset);
        self->m_dim_size = dim_size;
        self->m_stride = stride;
        self->m_dst_tp = dst_tp;
        self->m_dst_arrmeta = dst_arrmeta;
        self->m_dim_broadcast = dim_broadcast;
        // from pyobject ckernel
        ckb_offset = copy_from_pyobject::get().get()->instantiate(
            copy_from_pyobject::get().get()->static_data, 0, NULL, ckb,
            ckb_offset, el_tp, el_arrmeta, nsrc, src_tp, src_arrmeta,
            dynd::kernel_request_strided, ectx, nkwd, kwds, tp_vars);
        self = reinterpret_cast<
            dynd::ckernel_builder<dynd::kernel_request_host> *>(ckb)
                   ->get_at<pydynd::nd::copy_from_pyobject_kernel<
                         dynd::fixed_dim_type_id>>(root_ckb_offset);
        self->m_copy_dst_offset = ckb_offset - root_ckb_offset;
        // dst to dst ckernel, for broadcasting case
        return dynd::make_assignment_kernel(ckb, ckb_offset, el_tp, el_arrmeta,
                                            el_tp, el_arrmeta,
                                            dynd::kernel_request_strided, ectx);
      }

      throw std::runtime_error("could not process as strided");
    }
  };

  template <>
  struct copy_from_pyobject_kernel<
      dynd::var_dim_type_id> : dynd::nd::
                                   base_kernel<copy_from_pyobject_kernel<
                                                   dynd::var_dim_type_id>,
                                               1> {
    intptr_t m_offset, m_stride;
    dynd::ndt::type m_dst_tp;
    const char *m_dst_arrmeta;
    bool m_dim_broadcast;
    // Offset to ckernel which copies from dst to dst, for broadcasting case
    intptr_t m_copy_dst_offset;

    ~copy_from_pyobject_kernel()
    {
      get_child()->destroy();
    }

    inline void single(char *dst, char *const *src)
    {
      PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);

      if (DyND_PyArray_Check(src_obj)) {
        dynd::typed_data_assign(m_dst_tp, m_dst_arrmeta, dst,
                                ((DyND_PyArrayObject *)src_obj)->v);
        return;
      }
#ifdef DYND_NUMPY_INTEROP
      if (PyArray_Check(src_obj)) {
        array_copy_from_numpy(m_dst_tp, m_dst_arrmeta, dst,
                              (PyArrayObject *)src_obj,
                              &dynd::eval::default_eval_context);
        return;
      }
#endif
      // TODO: PEP 3118 support here

      ckernel_prefix *copy_el = get_child();
      dynd::expr_strided_t copy_el_fn =
          copy_el->get_function<dynd::expr_strided_t>();

      // Get the input as an array of PyObject *
      pyobject_ownref src_fast;
      char *child_src;
      intptr_t child_stride = sizeof(PyObject *);
      intptr_t src_dim_size;
      if (m_dim_broadcast && broadcast_as_scalar(m_dst_tp, src_obj)) {
        child_src = src[0];
        src_dim_size = 1;
      } else {
        src_fast.reset(PySequence_Fast(
            src_obj, "Require a sequence to copy to a dynd dimension"));
        child_src =
            reinterpret_cast<char *>(PySequence_Fast_ITEMS(src_fast.get()));
        src_dim_size = PySequence_Fast_GET_SIZE(src_fast.get());
      }

      // If the var dim element hasn't been allocated, initialize it
      dynd::var_dim_type_data *vdd =
          reinterpret_cast<dynd::var_dim_type_data *>(dst);
      if (vdd->begin == NULL) {
        if (m_offset != 0) {
          throw std::runtime_error(
              "Cannot assign to an uninitialized dynd var_dim "
              "which has a non-zero offset");
        }
        dynd::ndt::var_dim_element_initialize(m_dst_tp, m_dst_arrmeta, dst,
                                              src_dim_size);
      }

      if (src_dim_size != 1 && vdd->size != src_dim_size) {
        std::stringstream ss;
        ss << "Cannot assign python value " << pyobject_repr(src_obj)
           << " to a dynd " << m_dst_tp << " value";
        throw dynd::broadcast_error(ss.str());
      }
      if (src_dim_size == 1 && vdd->size > 1) {
        // Copy once from Python, then duplicate that element
        copy_el_fn(copy_el, vdd->begin + m_offset, 0, &child_src, &child_stride,
                   1);
        ckernel_prefix *copy_dst = get_child(m_copy_dst_offset);
        dynd::expr_strided_t copy_dst_fn =
            copy_dst->get_function<dynd::expr_strided_t>();
        intptr_t zero = 0;
        char *src_to_dup = vdd->begin + m_offset;
        copy_dst_fn(copy_dst, vdd->begin + m_offset + m_stride, m_stride,
                    &src_to_dup, &zero, vdd->size - 1);
      } else {
        copy_el_fn(copy_el, vdd->begin + m_offset, m_stride, &child_src,
                   &child_stride, vdd->size);
      }
      if (PyErr_Occurred()) {
        throw std::exception();
      }
    }

    static intptr_t
    instantiate(char *static_data, size_t data_size, char *data, void *ckb,
                intptr_t ckb_offset, const dynd::ndt::type &dst_tp,
                const char *dst_arrmeta, intptr_t nsrc,
                const dynd::ndt::type *src_tp, const char *const *src_arrmeta,
                dynd::kernel_request_t kernreq,
                const dynd::eval::eval_context *ectx, intptr_t nkwd,
                const dynd::nd::array *kwds,
                const std::map<std::string, dynd::ndt::type> &tp_vars)
    {
      bool dim_broadcast = kwds[0].as<bool>();

      intptr_t root_ckb_offset = ckb_offset;
      pydynd::nd::copy_from_pyobject_kernel<dynd::var_dim_type_id> *self =
          pydynd::nd::copy_from_pyobject_kernel<dynd::var_dim_type_id>::make(
              ckb, kernreq, ckb_offset);
      self->m_offset = reinterpret_cast<const dynd::var_dim_type_arrmeta *>(
          dst_arrmeta)->offset;
      self->m_stride = reinterpret_cast<const dynd::var_dim_type_arrmeta *>(
          dst_arrmeta)->stride;
      self->m_dst_tp = dst_tp;
      self->m_dst_arrmeta = dst_arrmeta;
      self->m_dim_broadcast = dim_broadcast;
      dynd::ndt::type el_tp =
          dst_tp.extended<dynd::ndt::var_dim_type>()->get_element_type();
      const char *el_arrmeta = dst_arrmeta + sizeof(dynd::var_dim_type_arrmeta);
      ckb_offset = copy_from_pyobject::get().get()->instantiate(
          copy_from_pyobject::get().get()->static_data, 0, NULL, ckb,
          ckb_offset, el_tp, el_arrmeta, nsrc, src_tp, src_arrmeta,
          dynd::kernel_request_strided, ectx, nkwd, kwds, tp_vars);
      self =
          reinterpret_cast<dynd::ckernel_builder<dynd::kernel_request_host> *>(
              ckb)
              ->get_at<pydynd::nd::copy_from_pyobject_kernel<
                    dynd::var_dim_type_id>>(root_ckb_offset);
      self->m_copy_dst_offset = ckb_offset - root_ckb_offset;
      // dst to dst ckernel, for broadcasting case
      return dynd::make_assignment_kernel(ckb, ckb_offset, el_tp, el_arrmeta,
                                          el_tp, el_arrmeta,
                                          dynd::kernel_request_strided, ectx);
    }
  };

  // TODO: Should make a more efficient strided kernel function
  template <>
  struct copy_from_pyobject_kernel<
      dynd::tuple_type_id> : dynd::nd::
                                 base_kernel<copy_from_pyobject_kernel<
                                                 dynd::tuple_type_id>,
                                             1> {
    dynd::ndt::type m_dst_tp;
    const char *m_dst_arrmeta;
    bool m_dim_broadcast;
    std::vector<intptr_t> m_copy_el_offsets;

    ~copy_from_pyobject_kernel()
    {
      for (size_t i = 0; i < m_copy_el_offsets.size(); ++i) {
        get_child(m_copy_el_offsets[i])->destroy();
      }
    }

    void single(char *dst, char *const *src)
    {
      PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);

      if (DyND_PyArray_Check(src_obj)) {
        dynd::typed_data_assign(m_dst_tp, m_dst_arrmeta, dst,
                                ((DyND_PyArrayObject *)src_obj)->v);
        return;
      }
#ifdef DYND_NUMPY_INTEROP
      if (PyArray_Check(src_obj)) {
        array_copy_from_numpy(m_dst_tp, m_dst_arrmeta, dst,
                              (PyArrayObject *)src_obj,
                              &dynd::eval::default_eval_context);
        return;
      }
#endif
      // TODO: PEP 3118 support here

      intptr_t field_count =
          m_dst_tp.extended<dynd::ndt::base_tuple_type>()->get_field_count();
      const uintptr_t *field_offsets =
          m_dst_tp.extended<dynd::ndt::base_tuple_type>()->get_data_offsets(
              m_dst_arrmeta);

      // Get the input as an array of PyObject *
      pyobject_ownref src_fast;
      char *child_src;
      intptr_t child_stride = sizeof(PyObject *);
      intptr_t src_dim_size;
      if (m_dim_broadcast && broadcast_as_scalar(m_dst_tp, src_obj)) {
        child_src = src[0];
        src_dim_size = 1;
      } else {
        src_fast.reset(PySequence_Fast(
            src_obj, "Require a sequence to copy to a dynd tuple"));
        child_src =
            reinterpret_cast<char *>(PySequence_Fast_ITEMS(src_fast.get()));
        src_dim_size = PySequence_Fast_GET_SIZE(src_fast.get());
      }

      if (src_dim_size != 1 && field_count != src_dim_size) {
        std::stringstream ss;
        ss << "Cannot assign python value " << pyobject_repr(src_obj)
           << " to a dynd " << m_dst_tp << " value";
        throw dynd::broadcast_error(ss.str());
      }
      if (src_dim_size == 1) {
        child_stride = 0;
      }
      for (intptr_t i = 0; i < field_count; ++i) {
        ckernel_prefix *copy_el = get_child(m_copy_el_offsets[i]);
        dynd::expr_single_t copy_el_fn =
            copy_el->get_function<dynd::expr_single_t>();
        char *el_dst = dst + field_offsets[i];
        char *el_src = child_src + i * child_stride;
        copy_el_fn(copy_el, el_dst, &el_src);
      }
      if (PyErr_Occurred()) {
        throw std::exception();
      }
    }

    static intptr_t
    instantiate(char *static_data, size_t data_size, char *data, void *ckb,
                intptr_t ckb_offset, const dynd::ndt::type &dst_tp,
                const char *dst_arrmeta, intptr_t nsrc,
                const dynd::ndt::type *src_tp, const char *const *src_arrmeta,
                dynd::kernel_request_t kernreq,
                const dynd::eval::eval_context *ectx, intptr_t nkwd,
                const dynd::nd::array *kwds,
                const std::map<std::string, dynd::ndt::type> &tp_vars)
    {
      bool dim_broadcast = kwds[0].as<bool>();

      intptr_t root_ckb_offset = ckb_offset;
      pydynd::nd::copy_from_pyobject_kernel<dynd::tuple_type_id> *self =
          pydynd::nd::copy_from_pyobject_kernel<dynd::tuple_type_id>::make(
              ckb, kernreq, ckb_offset);
      self->m_dst_tp = dst_tp;
      self->m_dst_arrmeta = dst_arrmeta;
      intptr_t field_count =
          dst_tp.extended<dynd::ndt::base_tuple_type>()->get_field_count();
      const dynd::ndt::type *field_types =
          dst_tp.extended<dynd::ndt::base_tuple_type>()->get_field_types_raw();
      const uintptr_t *arrmeta_offsets =
          dst_tp.extended<dynd::ndt::base_tuple_type>()
              ->get_arrmeta_offsets_raw();
      self->m_dim_broadcast = dim_broadcast;
      self->m_copy_el_offsets.resize(field_count);
      for (intptr_t i = 0; i < field_count; ++i) {
        reinterpret_cast<dynd::ckernel_builder<dynd::kernel_request_host> *>(
            ckb)->reserve(ckb_offset);
        self = reinterpret_cast<
            dynd::ckernel_builder<dynd::kernel_request_host> *>(ckb)
                   ->get_at<pydynd::nd::copy_from_pyobject_kernel<
                         dynd::tuple_type_id>>(root_ckb_offset);
        self->m_copy_el_offsets[i] = ckb_offset - root_ckb_offset;
        const char *field_arrmeta = dst_arrmeta + arrmeta_offsets[i];
        ckb_offset = copy_from_pyobject::get().get()->instantiate(
            copy_from_pyobject::get().get()->static_data, 0, NULL, ckb,
            ckb_offset, field_types[i], field_arrmeta, nsrc, src_tp,
            src_arrmeta, dynd::kernel_request_single, ectx, nkwd, kwds,
            tp_vars);
      }
      return ckb_offset;
    }
  };

  // TODO: Should make a more efficient strided kernel function
  template <>
  struct copy_from_pyobject_kernel<
      dynd::struct_type_id> : dynd::nd::
                                  base_kernel<copy_from_pyobject_kernel<
                                                  dynd::struct_type_id>,
                                              1> {
    dynd::ndt::type m_dst_tp;
    const char *m_dst_arrmeta;
    bool m_dim_broadcast;
    std::vector<intptr_t> m_copy_el_offsets;

    ~copy_from_pyobject_kernel()
    {
      for (size_t i = 0; i < m_copy_el_offsets.size(); ++i) {
        get_child(m_copy_el_offsets[i])->destroy();
      }
    }

    void single(char *dst, char *const *src)
    {
      PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);

      if (DyND_PyArray_Check(src_obj)) {
        dynd::typed_data_assign(m_dst_tp, m_dst_arrmeta, dst,
                                ((DyND_PyArrayObject *)src_obj)->v);
        return;
      }
#ifdef DYND_NUMPY_INTEROP
      if (PyArray_Check(src_obj)) {
        array_copy_from_numpy(m_dst_tp, m_dst_arrmeta, dst,
                              (PyArrayObject *)src_obj,
                              &dynd::eval::default_eval_context);
        return;
      }
#endif
      // TODO: PEP 3118 support here

      intptr_t field_count =
          m_dst_tp.extended<dynd::ndt::base_tuple_type>()->get_field_count();
      const uintptr_t *field_offsets =
          m_dst_tp.extended<dynd::ndt::base_tuple_type>()->get_data_offsets(
              m_dst_arrmeta);

      if (PyDict_Check(src_obj)) {
        // Keep track of which fields we've seen
        dynd::shortvector<bool> populated_fields(field_count);
        memset(populated_fields.get(), 0, sizeof(bool) * field_count);

        PyObject *dict_key = NULL, *dict_value = NULL;
        Py_ssize_t dict_pos = 0;

        while (PyDict_Next(src_obj, &dict_pos, &dict_key, &dict_value)) {
          std::string name = pydynd::pystring_as_string(dict_key);
          intptr_t i =
              m_dst_tp.extended<dynd::ndt::base_struct_type>()->get_field_index(
                  name);
          // TODO: Add an error policy of whether to throw an error
          //       or not. For now, just raise an error
          if (i >= 0) {
            ckernel_prefix *copy_el = get_child(m_copy_el_offsets[i]);
            dynd::expr_single_t copy_el_fn =
                copy_el->get_function<dynd::expr_single_t>();
            char *el_dst = dst + field_offsets[i];
            char *el_src = reinterpret_cast<char *>(&dict_value);
            copy_el_fn(copy_el, el_dst, &el_src);
            populated_fields[i] = true;
          } else {
            std::stringstream ss;
            ss << "Input python dict has key ";
            dynd::print_escaped_utf8_string(ss, name);
            ss << ", but no such field is in destination dynd type "
               << m_dst_tp;
            throw dynd::broadcast_error(ss.str());
          }
        }

        for (intptr_t i = 0; i < field_count; ++i) {
          if (!populated_fields[i]) {
            std::stringstream ss;
            ss << "python dict does not contain the field ";
            dynd::print_escaped_utf8_string(
                ss, m_dst_tp.extended<dynd::ndt::base_struct_type>()
                        ->get_field_name(i));
            ss << " as required by the data type " << m_dst_tp;
            throw dynd::broadcast_error(ss.str());
          }
        }
      } else {
        // Get the input as an array of PyObject *
        pyobject_ownref src_fast;
        char *child_src;
        intptr_t child_stride = sizeof(PyObject *);
        intptr_t src_dim_size;
        if (m_dim_broadcast && broadcast_as_scalar(m_dst_tp, src_obj)) {
          child_src = src[0];
          src_dim_size = 1;
        } else {
          src_fast.reset(PySequence_Fast(
              src_obj, "Require a sequence to copy to a dynd struct"));
          child_src =
              reinterpret_cast<char *>(PySequence_Fast_ITEMS(src_fast.get()));
          src_dim_size = PySequence_Fast_GET_SIZE(src_fast.get());
        }

        if (src_dim_size != 1 && field_count != src_dim_size) {
          std::stringstream ss;
          ss << "Cannot assign python value " << pyobject_repr(src_obj)
             << " to a dynd " << m_dst_tp << " value";
          throw dynd::broadcast_error(ss.str());
        }
        if (src_dim_size == 1) {
          child_stride = 0;
        }
        for (intptr_t i = 0; i < field_count; ++i) {
          ckernel_prefix *copy_el = get_child(m_copy_el_offsets[i]);
          dynd::expr_single_t copy_el_fn =
              copy_el->get_function<dynd::expr_single_t>();
          char *el_dst = dst + field_offsets[i];
          char *el_src = child_src + i * child_stride;
          copy_el_fn(copy_el, el_dst, &el_src);
        }
      }
      if (PyErr_Occurred()) {
        throw std::exception();
      }
    }

    static intptr_t
    instantiate(char *static_data, size_t data_size, char *data, void *ckb,
                intptr_t ckb_offset, const dynd::ndt::type &dst_tp,
                const char *dst_arrmeta, intptr_t nsrc,
                const dynd::ndt::type *src_tp, const char *const *src_arrmeta,
                dynd::kernel_request_t kernreq,
                const dynd::eval::eval_context *ectx, intptr_t nkwd,
                const dynd::nd::array *kwds,
                const std::map<std::string, dynd::ndt::type> &tp_vars)
    {
      bool dim_broadcast = kwds[0].as<bool>();

      intptr_t root_ckb_offset = ckb_offset;
      pydynd::nd::copy_from_pyobject_kernel<dynd::struct_type_id> *self =
          pydynd::nd::copy_from_pyobject_kernel<dynd::struct_type_id>::make(
              ckb, kernreq, ckb_offset);
      self->m_dst_tp = dst_tp;
      self->m_dst_arrmeta = dst_arrmeta;
      intptr_t field_count =
          dst_tp.extended<dynd::ndt::base_struct_type>()->get_field_count();
      const dynd::ndt::type *field_types =
          dst_tp.extended<dynd::ndt::base_struct_type>()->get_field_types_raw();
      const uintptr_t *arrmeta_offsets =
          dst_tp.extended<dynd::ndt::base_struct_type>()
              ->get_arrmeta_offsets_raw();
      self->m_dim_broadcast = dim_broadcast;
      self->m_copy_el_offsets.resize(field_count);
      for (intptr_t i = 0; i < field_count; ++i) {
        reinterpret_cast<dynd::ckernel_builder<dynd::kernel_request_host> *>(
            ckb)->reserve(ckb_offset);
        self = reinterpret_cast<
            dynd::ckernel_builder<dynd::kernel_request_host> *>(ckb)
                   ->get_at<pydynd::nd::copy_from_pyobject_kernel<
                         dynd::struct_type_id>>(root_ckb_offset);
        self->m_copy_el_offsets[i] = ckb_offset - root_ckb_offset;
        const char *field_arrmeta = dst_arrmeta + arrmeta_offsets[i];
        ckb_offset = copy_from_pyobject::get().get()->instantiate(
            copy_from_pyobject::get().get()->static_data, 0, NULL, ckb,
            ckb_offset, field_types[i], field_arrmeta, nsrc, src_tp,
            src_arrmeta, dynd::kernel_request_single, ectx, nkwd, kwds,
            tp_vars);
      }
      return ckb_offset;
    }
  };

  struct default_copy_from_pyobject_kernel
      : dynd::nd::base_virtual_kernel<default_copy_from_pyobject_kernel> {
    static intptr_t
    instantiate(char *static_data, size_t data_size, char *data, void *ckb,
                intptr_t ckb_offset, const dynd::ndt::type &dst_tp,
                const char *dst_arrmeta, intptr_t nsrc,
                const dynd::ndt::type *src_tp, const char *const *src_arrmeta,
                dynd::kernel_request_t kernreq,
                const dynd::eval::eval_context *ectx, intptr_t nkwd,
                const dynd::nd::array *kwds,
                const std::map<std::string, dynd::ndt::type> &tp_vars)
    {
      if (dst_tp.get_kind() == dynd::expr_kind) {
        dynd::nd::callable af = dynd::nd::functional::compose(
            copy_from_pyobject, dynd::nd::copy, dst_tp.value_type());
        return af.get()->instantiate(af.get()->static_data, 0, NULL, ckb,
                                     ckb_offset, dst_tp, dst_arrmeta, nsrc,
                                     src_tp, src_arrmeta, kernreq, ectx, nkwd,
                                     kwds, tp_vars);
      }

      std::stringstream ss;
      ss << "Unable to copy a Python object to dynd value with type " << dst_tp;
      throw std::invalid_argument(ss.str());
    }
  };

} // namespace pydynd::nd
} // namespace pydynd

namespace dynd {
namespace ndt {

  template <>
  struct type::equivalent<pydynd::nd::copy_from_pyobject_kernel<bool_type_id>> {
    static type make()
    {
      return type("(void) -> bool");
    }
  };

  template <>
  struct type::equivalent<pydynd::nd::copy_from_pyobject_kernel<int8_type_id>> {
    static type make()
    {
      return type("(void) -> int8");
    }
  };

  template <>
  struct type::equivalent<
      pydynd::nd::copy_from_pyobject_kernel<int16_type_id>> {
    static type make()
    {
      return type("(void) -> int16");
    }
  };

  template <>
  struct type::equivalent<
      pydynd::nd::copy_from_pyobject_kernel<int32_type_id>> {
    static type make()
    {
      return type("(void) -> int32");
    }
  };

  template <>
  struct type::equivalent<
      pydynd::nd::copy_from_pyobject_kernel<int64_type_id>> {
    static type make()
    {
      return type("(void) -> int64");
    }
  };

  template <>
  struct type::equivalent<
      pydynd::nd::copy_from_pyobject_kernel<int128_type_id>> {
    static type make()
    {
      return type("(void) -> int128");
    }
  };

  template <>
  struct type::equivalent<
      pydynd::nd::copy_from_pyobject_kernel<uint8_type_id>> {
    static type make()
    {
      return type("(void) -> uint8");
    }
  };

  template <>
  struct type::equivalent<
      pydynd::nd::copy_from_pyobject_kernel<uint16_type_id>> {
    static type make()
    {
      return type("(void) -> uint16");
    }
  };

  template <>
  struct type::equivalent<
      pydynd::nd::copy_from_pyobject_kernel<uint32_type_id>> {
    static type make()
    {
      return type("(void) -> uint32");
    }
  };

  template <>
  struct type::equivalent<
      pydynd::nd::copy_from_pyobject_kernel<uint64_type_id>> {
    static type make()
    {
      return type("(void) -> uint64");
    }
  };

  template <>
  struct type::equivalent<
      pydynd::nd::copy_from_pyobject_kernel<uint128_type_id>> {
    static type make()
    {
      return type("(void) -> uint128");
    }
  };

  template <>
  struct type::equivalent<
      pydynd::nd::copy_from_pyobject_kernel<float16_type_id>> {
    static type make()
    {
      return type("(void) -> float16");
    }
  };

  template <>
  struct type::equivalent<
      pydynd::nd::copy_from_pyobject_kernel<float32_type_id>> {
    static type make()
    {
      return type("(void) -> float32");
    }
  };

  template <>
  struct type::equivalent<
      pydynd::nd::copy_from_pyobject_kernel<float64_type_id>> {
    static type make()
    {
      return type("(void) -> float64");
    }
  };

  template <>
  struct type::equivalent<
      pydynd::nd::copy_from_pyobject_kernel<complex_float32_type_id>> {
    static type make()
    {
      return type("(void) -> complex[float32]");
    }
  };

  template <>
  struct type::equivalent<
      pydynd::nd::copy_from_pyobject_kernel<complex_float64_type_id>> {
    static type make()
    {
      return type("(void) -> complex[float64]");
    }
  };

  template <>
  struct type::equivalent<
      pydynd::nd::copy_from_pyobject_kernel<bytes_type_id>> {
    static type make()
    {
      return type("(void) -> bytes");
    }
  };

  template <>
  struct type::equivalent<
      pydynd::nd::copy_from_pyobject_kernel<fixed_bytes_type_id>> {
    static type make()
    {
      return type("(void) -> FixedBytes");
    }
  };

  template <>
  struct type::equivalent<
      pydynd::nd::copy_from_pyobject_kernel<string_type_id>> {
    static type make()
    {
      return type("(void) -> string");
    }
  };

  template <>
  struct type::equivalent<
      pydynd::nd::copy_from_pyobject_kernel<fixed_string_type_id>> {
    static type make()
    {
      return type("(void) -> FixedString");
    }
  };

  template <>
  struct type::equivalent<pydynd::nd::copy_from_pyobject_kernel<date_type_id>> {
    static type make()
    {
      return type("(void) -> date");
    }
  };

  template <>
  struct type::equivalent<pydynd::nd::copy_from_pyobject_kernel<time_type_id>> {
    static type make()
    {
      return type("(void) -> time");
    }
  };

  template <>
  struct type::equivalent<
      pydynd::nd::copy_from_pyobject_kernel<datetime_type_id>> {
    static type make()
    {
      return type("(void) -> datetime");
    }
  };

  template <>
  struct type::equivalent<pydynd::nd::copy_from_pyobject_kernel<type_type_id>> {
    static type make()
    {
      return type("(void) -> type");
    }
  };

  template <>
  struct type::equivalent<
      pydynd::nd::copy_from_pyobject_kernel<option_type_id>> {
    static type make()
    {
      return type("(void) -> ?Any");
    }
  };

  template <>
  struct type::equivalent<
      pydynd::nd::copy_from_pyobject_kernel<categorical_type_id>> {
    static type make()
    {
      return type("(void) -> Categorical");
    }
  };

  template <>
  struct type::equivalent<
      pydynd::nd::copy_from_pyobject_kernel<fixed_dim_type_id>> {
    static type make()
    {
      return type("(void, broadcast: bool) -> Fixed * Any");
    }
  };

  template <>
  struct type::equivalent<
      pydynd::nd::copy_from_pyobject_kernel<var_dim_type_id>> {
    static type make()
    {
      return type("(void, broadcast: bool) -> var * Any");
    }
  };

  template <>
  struct type::equivalent<
      pydynd::nd::copy_from_pyobject_kernel<tuple_type_id>> {
    static type make()
    {
      return type("(void, broadcast: bool) -> (...)");
    }
  };

  template <>
  struct type::equivalent<
      pydynd::nd::copy_from_pyobject_kernel<struct_type_id>> {
    static type make()
    {
      return type("(void, broadcast: bool) -> {...}");
    }
  };

  template <>
  struct type::equivalent<pydynd::nd::default_copy_from_pyobject_kernel> {
    static type make()
    {
      return type("(void, broadcast: bool) -> Any");
    }
  };

} // namespace dynd::ndt
} // namespace dynd

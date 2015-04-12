#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/base_virtual_kernel.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/parser_util.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/datetime_type.hpp>
#include <dynd/types/time_type.hpp>
#include <dynd/types/var_dim_type.hpp>

#include "config.hpp"

namespace pydynd {
namespace nd {

  template <type_id_t src_type_id>
  struct copy_from_pyobject_kernel;

  template <>
  struct copy_from_pyobject_kernel<bool_type_id>
      : base_kernel<copy_from_pyobject_kernel<bool_type_id>,
                    kernel_request_host, 1> {
    void single(char *dst, char *const *src)
    {
      PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);
      if (src_obj == Py_True) {
        *dst = 1;
      } else if (src_obj == Py_False) {
        *dst = 0;
      } else {
        *dst = array_from_py(src_obj, 0, false, &eval::default_eval_context)
                   .as<dynd_bool>();
      }
    }
  };

  void pyint_to_int(int8_t *out, PyObject *obj)
  {
    long v = PyLong_AsLong(obj);
    if (v == -1 && PyErr_Occurred()) {
      throw std::exception();
    }
    if (parse::overflow_check<int8_t>::is_overflow(v)) {
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
    if (parse::overflow_check<int16_t>::is_overflow(v)) {
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
    if (parse::overflow_check<int32_t>::is_overflow(v)) {
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

  void pyint_to_int(dynd_int128 *out, PyObject *obj)
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
    dynd_int128 result(hi, lo);

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
    if (parse::overflow_check<uint8_t>::is_overflow(v)) {
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
    if (parse::overflow_check<uint16_t>::is_overflow(v)) {
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
    if (parse::overflow_check<uint32_t>::is_overflow(v)) {
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

  void pyint_to_int(dynd_uint128 *out, PyObject *obj)
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
    dynd_uint128 result(hi, lo);

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
  struct int_copy_from_pyobject_kernel
      : base_kernel<int_copy_from_pyobject_kernel<T>, kernel_request_host, 1> {
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
        *reinterpret_cast<T *>(dst) =
            array_from_py(src_obj, 0, false, &eval::default_eval_context)
                .as<T>();
      }
    }
  };

  template <typename T>
  struct float_copy_from_pyobject_kernel
      : base_kernel<float_copy_from_pyobject_kernel<T>, kernel_request_host,
                    1> {
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
        *reinterpret_cast<T *>(dst) =
            array_from_py(src_obj, 0, false, &eval::default_eval_context)
                .as<T>();
      }
    }
  };

  template <typename T>
  struct complex_float_copy_from_pyobject_kernel
      : base_kernel<complex_float_copy_from_pyobject_kernel<T>,
                    kernel_request_host, 1> {
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
            array_from_py(src_obj, 0, false, &eval::default_eval_context)
                .as<dynd::complex<T>>();
      }
    }
  };

  // TODO: This is not very efficient, could be made better
  //       with specialized bytes/fixed_bytes versions
  struct bytes_copy_from_pyobject_kernel
      : base_kernel<bytes_copy_from_pyobject_kernel, kernel_request_host, 1> {
    ndt::type dst_tp;
    const char *dst_arrmeta;

    bytes_copy_from_pyobject_kernel(const ndt::type &dst_tp,
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
      } else if (WArray_Check(src_obj)) {
        typed_data_assign(dst_tp, dst_arrmeta, dst, ((WArray *)src_obj)->v);
        return;
      } else {
        std::stringstream ss;
        ss << "Cannot assign object " << pyobject_repr(src_obj)
           << " to a dynd bytes value";
        throw std::invalid_argument(ss.str());
      }

      ndt::type bytes_tp = ndt::make_bytes(1);
      string_type_data bytes_d;
      string_type_arrmeta bytes_md;
      bytes_d.begin = pybytes_data;
      bytes_d.end = pybytes_data + pybytes_len;
      bytes_md.blockref = NULL;

      typed_data_assign(dst_tp, dst_arrmeta, dst, bytes_tp,
                        reinterpret_cast<const char *>(&bytes_md),
                        reinterpret_cast<const char *>(&bytes_d));
    }
  };

  // TODO: This is not very efficient, could be made better
  //       with specialized versions
  struct string_copy_from_pyobject_kernel
      : base_kernel<string_copy_from_pyobject_kernel, kernel_request_host, 1> {
    ndt::type dst_tp;
    const char *dst_arrmeta;

    string_copy_from_pyobject_kernel(const ndt::type &dst_tp,
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

        ndt::type str_tp = ndt::make_string();
        string_type_data str_d;
        string_type_arrmeta str_md;
        str_d.begin = s;
        str_d.end = s + len;
        str_md.blockref = NULL;

        typed_data_assign(dst_tp, dst_arrmeta, dst, str_tp,
                          reinterpret_cast<const char *>(&str_md),
                          reinterpret_cast<const char *>(&str_d));
#if PY_VERSION_HEX < 0x03000000
      } else if (PyString_Check(src_obj)) {
        char *pystr_data = NULL;
        intptr_t pystr_len = 0;
        if (PyString_AsStringAndSize(src_obj, &pystr_data, &pystr_len) < 0) {
          throw std::runtime_error("Error getting string data");
        }

        ndt::type str_dt = ndt::make_string(string_encoding_ascii);
        string_type_data str_d;
        string_type_arrmeta str_md;
        str_d.begin = pystr_data;
        str_d.end = pystr_data + pystr_len;
        str_md.blockref = NULL;

        typed_data_assign(dst_tp, dst_arrmeta, dst, str_dt,
                          reinterpret_cast<const char *>(&str_md),
                          reinterpret_cast<const char *>(&str_d));
#endif
      } else if (WArray_Check(src_obj)) {
        typed_data_assign(dst_tp, dst_arrmeta, dst, ((WArray *)src_obj)->v);
        return;
      } else {
        std::stringstream ss;
        ss << "Cannot assign object " << pyobject_repr(src_obj)
           << " to a dynd bytes value";
        throw std::invalid_argument(ss.str());
      }
    }
  };

  template <>
  struct copy_from_pyobject_kernel<date_type_id>
      : base_kernel<copy_from_pyobject_kernel<date_type_id>,
                    kernel_request_host, 1> {
    dynd::ndt::type dst_tp;
    const char *dst_arrmeta;

    copy_from_pyobject_kernel(const ndt::type &dst_tp, const char *dst_arrmeta)
        : dst_tp(dst_tp), dst_arrmeta(dst_arrmeta)
    {
    }

    void single(char *dst, char *const *src)
    {
      PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);
      if (PyDate_Check(src_obj)) {
        const date_type *dd = dst_tp.extended<date_type>();
        dd->set_ymd(dst_arrmeta, dst, assign_error_fractional,
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
        const date_type *dd = dst_tp.extended<date_type>();
        dd->set_ymd(dst_arrmeta, dst, assign_error_fractional,
                    PyDateTime_GET_YEAR(src_obj), PyDateTime_GET_MONTH(src_obj),
                    PyDateTime_GET_DAY(src_obj));
      } else if (WArray_Check(src_obj)) {
        typed_data_assign(dst_tp, dst_arrmeta, dst, ((WArray *)src_obj)->v);
      } else {
        typed_data_assign(
            dst_tp, dst_arrmeta, dst,
            array_from_py(src_obj, 0, false, &eval::default_eval_context));
      }
    }
  };

  template <>
  struct copy_from_pyobject_kernel<time_type_id>
      : base_kernel<copy_from_pyobject_kernel<time_type_id>,
                    kernel_request_host, 1> {

    dynd::ndt::type dst_tp;
    const char *dst_arrmeta;

    copy_from_pyobject_kernel(const ndt::type &dst_tp, const char *dst_arrmeta)
        : dst_tp(dst_tp), dst_arrmeta(dst_arrmeta)
    {
    }

    void single(char *dst, char *const *src)
    {
      PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);
      if (PyTime_Check(src_obj)) {
        const time_type *tt = dst_tp.extended<time_type>();
        tt->set_time(dst_arrmeta, dst, assign_error_fractional,
                     PyDateTime_TIME_GET_HOUR(src_obj),
                     PyDateTime_TIME_GET_MINUTE(src_obj),
                     PyDateTime_TIME_GET_SECOND(src_obj),
                     PyDateTime_TIME_GET_MICROSECOND(src_obj) *
                         DYND_TICKS_PER_MICROSECOND);
      } else if (WArray_Check(src_obj)) {
        typed_data_assign(dst_tp, dst_arrmeta, dst, ((WArray *)src_obj)->v);
      } else {
        typed_data_assign(
            dst_tp, dst_arrmeta, dst,
            array_from_py(src_obj, 0, false, &eval::default_eval_context));
      }
    }
  };

  template <>
  struct copy_from_pyobject_kernel<datetime_type_id>
      : base_kernel<copy_from_pyobject_kernel<datetime_type_id>,
                    kernel_request_host, 1> {
    dynd::ndt::type dst_tp;
    const char *dst_arrmeta;

    copy_from_pyobject_kernel(const ndt::type &dst_tp, const char *dst_arrmeta)
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
        const datetime_type *dd = dst_tp.extended<datetime_type>();
        dd->set_cal(dst_arrmeta, dst, assign_error_fractional,
                    PyDateTime_GET_YEAR(src_obj), PyDateTime_GET_MONTH(src_obj),
                    PyDateTime_GET_DAY(src_obj),
                    PyDateTime_DATE_GET_HOUR(src_obj),
                    PyDateTime_DATE_GET_MINUTE(src_obj),
                    PyDateTime_DATE_GET_SECOND(src_obj),
                    PyDateTime_DATE_GET_MICROSECOND(src_obj) * 10);
      } else if (WArray_Check(src_obj)) {
        typed_data_assign(dst_tp, dst_arrmeta, dst, ((WArray *)src_obj)->v);
      } else {
        typed_data_assign(
            dst_tp, dst_arrmeta, dst,
            array_from_py(src_obj, 0, false, &eval::default_eval_context));
      }
    }
  };

  template <>
  struct copy_from_pyobject_kernel<type_type_id>
      : base_kernel<copy_from_pyobject_kernel<type_type_id>,
                    kernel_request_host, 1> {
    void single(char *dst, char *const *src)
    {
      PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);
      *reinterpret_cast<dynd::ndt::type *>(dst) =
          make_ndt_type_from_pyobject(src_obj);
    }
  };

  // TODO: Should make a more efficient strided kernel function
  template <>
  struct copy_from_pyobject_kernel<option_type_id>
      : base_kernel<copy_from_pyobject_kernel<option_type_id>,
                    kernel_request_host, 1> {
    intptr_t m_copy_value_offset;
    dynd::ndt::type m_dst_tp;
    const char *m_dst_arrmeta;

    void single(char *dst, char *const *src)
    {
      PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);
      if (src_obj == Py_None) {
        ckernel_prefix *assign_na = get_child_ckernel();
        expr_single_t assign_na_fn = assign_na->get_function<expr_single_t>();
        assign_na_fn(dst, NULL, assign_na);
      } else if (WArray_Check(src_obj)) {
        typed_data_assign(m_dst_tp, m_dst_arrmeta, dst, ((WArray *)src_obj)->v);
      } else if (m_dst_tp.get_kind() != string_kind &&
                 PyUnicode_Check(src_obj)) {
        // Copy from the string
        pyobject_ownref utf8(PyUnicode_AsUTF8String(src_obj));
        char *s = NULL;
        Py_ssize_t len = 0;
        if (PyBytes_AsStringAndSize(utf8.get(), &s, &len) < 0) {
          throw std::exception();
        }

        dynd::ndt::type str_tp = dynd::ndt::make_string();
        string_type_arrmeta str_md;
        string_type_data str_d;
        str_d.begin = s;
        str_d.end = s + len;
        const char *src_str = reinterpret_cast<const char *>(&str_d);
        str_md.blockref = NULL;

        typed_data_assign(m_dst_tp, m_dst_arrmeta, dst, str_tp,
                          reinterpret_cast<const char *>(&str_md),
                          reinterpret_cast<const char *>(&str_d));
#if PY_VERSION_HEX < 0x03000000
      } else if (m_dst_tp.get_kind() != string_kind &&
                 PyString_Check(src_obj)) {
        // Copy from the string
        char *s = NULL;
        Py_ssize_t len = 0;
        if (PyString_AsStringAndSize(src_obj, &s, &len) < 0) {
          throw std::exception();
        }

        dynd::ndt::type str_tp = dynd::ndt::make_string();
        string_type_arrmeta str_md;
        string_type_data str_d;
        str_d.begin = s;
        str_d.end = s + len;
        const char *src_str = reinterpret_cast<const char *>(&str_d);
        str_md.blockref = NULL;

        typed_data_assign(m_dst_tp, m_dst_arrmeta, dst, str_tp,
                          reinterpret_cast<const char *>(&str_md),
                          reinterpret_cast<const char *>(&str_d));
#endif
      } else {
        ckernel_prefix *copy_value = get_child_ckernel(m_copy_value_offset);
        expr_single_t copy_value_fn = copy_value->get_function<expr_single_t>();
        copy_value_fn(dst, src, copy_value);
      }
    }

    void destruct_children()
    {
      get_child_ckernel()->destroy();
      base.destroy_child_ckernel(m_copy_value_offset);
    }
  };

  // TODO: Could instantiate the dst_tp -> dst_tp assignment
  //       as part of the ckernel instead of dynamically
  template <>
  struct copy_from_pyobject_kernel<fixed_dim_type_id>
      : base_kernel<copy_from_pyobject_kernel<fixed_dim_type_id>,
                    kernel_request_host, 1> {
    intptr_t m_dim_size, m_stride;
    dynd::ndt::type m_dst_tp;
    const char *m_dst_arrmeta;
    bool m_dim_broadcast;
    // Offset to ckernel which copies from dst to dst, for broadcasting case
    intptr_t m_copy_dst_offset;

    inline void single(char *dst, char *const *src)
    {
      PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);

      if (WArray_Check(src_obj)) {
        typed_data_assign(m_dst_tp, m_dst_arrmeta, dst, ((WArray *)src_obj)->v);
        return;
      }
#ifdef DYND_NUMPY_INTEROP
      if (PyArray_Check(src_obj)) {
        array_copy_from_numpy(m_dst_tp, m_dst_arrmeta, dst,
                              (PyArrayObject *)src_obj,
                              &eval::default_eval_context);
        return;
      }
#endif
      // TODO: PEP 3118 support here

      ckernel_prefix *copy_el = get_child_ckernel();
      expr_strided_t copy_el_fn = copy_el->get_function<expr_strided_t>();

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
        throw broadcast_error(ss.str());
      }
      if (src_dim_size == 1 && m_dim_size > 1) {
        // Copy once from Python, then duplicate that element
        copy_el_fn(dst, 0, &child_src, &child_stride, 1, copy_el);
        ckernel_prefix *copy_dst = get_child_ckernel(m_copy_dst_offset);
        expr_strided_t copy_dst_fn = copy_dst->get_function<expr_strided_t>();
        intptr_t zero = 0;
        copy_dst_fn(dst + m_stride, m_stride, &dst, &zero, m_dim_size - 1,
                    copy_dst);
      } else {
        copy_el_fn(dst, m_stride, &child_src, &child_stride, m_dim_size,
                   copy_el);
      }
      if (PyErr_Occurred()) {
        throw std::exception();
      }
    }

    inline void destruct_children() { get_child_ckernel()->destroy(); }
  };

  template <>
  struct copy_from_pyobject_kernel<var_dim_type_id>
      : base_kernel<copy_from_pyobject_kernel<var_dim_type_id>,
                    kernel_request_host, 1> {
    intptr_t m_offset, m_stride;
    dynd::ndt::type m_dst_tp;
    const char *m_dst_arrmeta;
    bool m_dim_broadcast;
    // Offset to ckernel which copies from dst to dst, for broadcasting case
    intptr_t m_copy_dst_offset;

    inline void single(char *dst, char *const *src)
    {
      PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);

      if (WArray_Check(src_obj)) {
        typed_data_assign(m_dst_tp, m_dst_arrmeta, dst, ((WArray *)src_obj)->v);
        return;
      }
#ifdef DYND_NUMPY_INTEROP
      if (PyArray_Check(src_obj)) {
        array_copy_from_numpy(m_dst_tp, m_dst_arrmeta, dst,
                              (PyArrayObject *)src_obj,
                              &eval::default_eval_context);
        return;
      }
#endif
      // TODO: PEP 3118 support here

      ckernel_prefix *copy_el = get_child_ckernel();
      expr_strided_t copy_el_fn = copy_el->get_function<expr_strided_t>();

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
      var_dim_type_data *vdd = reinterpret_cast<var_dim_type_data *>(dst);
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
        throw broadcast_error(ss.str());
      }
      if (src_dim_size == 1 && vdd->size > 1) {
        // Copy once from Python, then duplicate that element
        copy_el_fn(vdd->begin + m_offset, 0, &child_src, &child_stride, 1,
                   copy_el);
        ckernel_prefix *copy_dst = get_child_ckernel(m_copy_dst_offset);
        expr_strided_t copy_dst_fn = copy_dst->get_function<expr_strided_t>();
        intptr_t zero = 0;
        char *src_to_dup = vdd->begin + m_offset;
        copy_dst_fn(vdd->begin + m_offset + m_stride, m_stride, &src_to_dup,
                    &zero, vdd->size - 1, copy_dst);
      } else {
        copy_el_fn(vdd->begin + m_offset, m_stride, &child_src, &child_stride,
                   vdd->size, copy_el);
      }
      if (PyErr_Occurred()) {
        throw std::exception();
      }
    }

    void destruct_children() { get_child_ckernel()->destroy(); }
  };

  // TODO: Should make a more efficient strided kernel function
  template <>
  struct copy_from_pyobject_kernel<tuple_type_id>
      : base_kernel<copy_from_pyobject_kernel<tuple_type_id>,
                    kernel_request_host, 1> {
    dynd::ndt::type m_dst_tp;
    const char *m_dst_arrmeta;
    bool m_dim_broadcast;
    std::vector<intptr_t> m_copy_el_offsets;

    void single(char *dst, char *const *src)
    {
      PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);

      if (WArray_Check(src_obj)) {
        typed_data_assign(m_dst_tp, m_dst_arrmeta, dst, ((WArray *)src_obj)->v);
        return;
      }
#ifdef DYND_NUMPY_INTEROP
      if (PyArray_Check(src_obj)) {
        array_copy_from_numpy(m_dst_tp, m_dst_arrmeta, dst,
                              (PyArrayObject *)src_obj,
                              &eval::default_eval_context);
        return;
      }
#endif
      // TODO: PEP 3118 support here

      intptr_t field_count =
          m_dst_tp.extended<base_tuple_type>()->get_field_count();
      const uintptr_t *field_offsets =
          m_dst_tp.extended<base_tuple_type>()->get_data_offsets(m_dst_arrmeta);

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
        throw broadcast_error(ss.str());
      }
      if (src_dim_size == 1) {
        child_stride = 0;
      }
      for (intptr_t i = 0; i < field_count; ++i) {
        ckernel_prefix *copy_el = get_child_ckernel(m_copy_el_offsets[i]);
        expr_single_t copy_el_fn = copy_el->get_function<expr_single_t>();
        char *el_dst = dst + field_offsets[i];
        char *el_src = child_src + i * child_stride;
        copy_el_fn(el_dst, &el_src, copy_el);
      }
      if (PyErr_Occurred()) {
        throw std::exception();
      }
    }

    void destruct_children()
    {
      for (size_t i = 0; i < m_copy_el_offsets.size(); ++i) {
        base.destroy_child_ckernel(m_copy_el_offsets[i]);
      }
    }
  };

  // TODO: Should make a more efficient strided kernel function
  template <>
  struct copy_from_pyobject_kernel<struct_type_id>
      : base_kernel<copy_from_pyobject_kernel<struct_type_id>, kernel_request_host, 1> {
    dynd::ndt::type m_dst_tp;
    const char *m_dst_arrmeta;
    bool m_dim_broadcast;
    std::vector<intptr_t> m_copy_el_offsets;

    void single(char *dst, char *const *src)
    {
      PyObject *src_obj = *reinterpret_cast<PyObject *const *>(src[0]);

      if (WArray_Check(src_obj)) {
        typed_data_assign(m_dst_tp, m_dst_arrmeta, dst, ((WArray *)src_obj)->v);
        return;
      }
#ifdef DYND_NUMPY_INTEROP
      if (PyArray_Check(src_obj)) {
        array_copy_from_numpy(m_dst_tp, m_dst_arrmeta, dst,
                              (PyArrayObject *)src_obj,
                              &eval::default_eval_context);
        return;
      }
#endif
      // TODO: PEP 3118 support here

      intptr_t field_count =
          m_dst_tp.extended<base_tuple_type>()->get_field_count();
      const uintptr_t *field_offsets =
          m_dst_tp.extended<base_tuple_type>()->get_data_offsets(m_dst_arrmeta);

      if (PyDict_Check(src_obj)) {
        // Keep track of which fields we've seen
        shortvector<bool> populated_fields(field_count);
        memset(populated_fields.get(), 0, sizeof(bool) * field_count);

        PyObject *dict_key = NULL, *dict_value = NULL;
        Py_ssize_t dict_pos = 0;

        while (PyDict_Next(src_obj, &dict_pos, &dict_key, &dict_value)) {
          std::string name = pystring_as_string(dict_key);
          intptr_t i =
              m_dst_tp.extended<base_struct_type>()->get_field_index(name);
          // TODO: Add an error policy of whether to throw an error
          //       or not. For now, just raise an error
          if (i >= 0) {
            ckernel_prefix *copy_el = get_child_ckernel(m_copy_el_offsets[i]);
            expr_single_t copy_el_fn = copy_el->get_function<expr_single_t>();
            char *el_dst = dst + field_offsets[i];
            char *el_src = reinterpret_cast<char *>(&dict_value);
            copy_el_fn(el_dst, &el_src, copy_el);
            populated_fields[i] = true;
          } else {
            std::stringstream ss;
            ss << "Input python dict has key ";
            print_escaped_utf8_string(ss, name);
            ss << ", but no such field is in destination dynd type "
               << m_dst_tp;
            throw broadcast_error(ss.str());
          }
        }

        for (intptr_t i = 0; i < field_count; ++i) {
          if (!populated_fields[i]) {
            std::stringstream ss;
            ss << "python dict does not contain the field ";
            print_escaped_utf8_string(
                ss, m_dst_tp.extended<base_struct_type>()->get_field_name(i));
            ss << " as required by the data type " << m_dst_tp;
            throw broadcast_error(ss.str());
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
          throw broadcast_error(ss.str());
        }
        if (src_dim_size == 1) {
          child_stride = 0;
        }
        for (intptr_t i = 0; i < field_count; ++i) {
          ckernel_prefix *copy_el = get_child_ckernel(m_copy_el_offsets[i]);
          expr_single_t copy_el_fn = copy_el->get_function<expr_single_t>();
          char *el_dst = dst + field_offsets[i];
          char *el_src = child_src + i * child_stride;
          copy_el_fn(el_dst, &el_src, copy_el);
        }
      }
      if (PyErr_Occurred()) {
        throw std::exception();
      }
    }

    inline void destruct_children()
    {
      for (size_t i = 0; i < m_copy_el_offsets.size(); ++i) {
        base.destroy_child_ckernel(m_copy_el_offsets[i]);
      }
    }
  };

} // namespace pydynd::nd
} // namespace pydynd
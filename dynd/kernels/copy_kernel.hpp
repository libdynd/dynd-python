#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/base_virtual_kernel.hpp>

#include "config.hpp"

namespace pydynd {
namespace nd {

  template <type_id_t src_type_id>
  struct copy_kernel;

  template <>
  struct copy_kernel<bool_type_id>
      : base_kernel<copy_kernel<bool_type_id>, kernel_request_host, 1> {
    void single(char *dst, char *const *src)
    {
      PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
      Py_XDECREF(*dst_obj);
      *dst_obj = (*src[0] != 0) ? Py_True : Py_False;
      Py_INCREF(*dst_obj);
    }
  };

  PyObject *pyint_from_int(int8_t v)
  {
#if PY_VERSION_HEX >= 0x03000000
    return PyLong_FromLong(v);
#else
    return PyInt_FromLong(v);
#endif
  }

  PyObject *pyint_from_int(uint8_t v)
  {
#if PY_VERSION_HEX >= 0x03000000
    return PyLong_FromLong(v);
#else
    return PyInt_FromLong(v);
#endif
  }

  PyObject *pyint_from_int(int16_t v)
  {
#if PY_VERSION_HEX >= 0x03000000
    return PyLong_FromLong(v);
#else
    return PyInt_FromLong(v);
#endif
  }

  PyObject *pyint_from_int(uint16_t v)
  {
#if PY_VERSION_HEX >= 0x03000000
    return PyLong_FromLong(v);
#else
    return PyInt_FromLong(v);
#endif
  }

  PyObject *pyint_from_int(int32_t v)
  {
#if PY_VERSION_HEX >= 0x03000000
    return PyLong_FromLong(v);
#else
    return PyInt_FromLong(v);
#endif
  }

  PyObject *pyint_from_int(uint32_t v) { return PyLong_FromUnsignedLong(v); }

#if SIZEOF_LONG == 8
  PyObject *pyint_from_int(int64_t v)
  {
#if PY_VERSION_HEX >= 0x03000000
    return PyLong_FromLong(v);
#else
    return PyInt_FromLong(v);
#endif
  }

  PyObject *pyint_from_int(uint64_t v) { return PyLong_FromUnsignedLong(v); }
#else
  PyObject *pyint_from_int(int64_t v) { return PyLong_FromLongLong(v); }

  PyObject *pyint_from_int(uint64_t v)
  {
    return PyLong_FromUnsignedLongLong(v);
  }
#endif

  PyObject *pyint_from_int(const dynd::dynd_uint128 &val)
  {
    if (val.m_hi == 0ULL) {
      return PyLong_FromUnsignedLongLong(val.m_lo);
    }
    // Use the pynumber methods to shift and or together the 64 bit parts
    pyobject_ownref hi(PyLong_FromUnsignedLongLong(val.m_hi));
    pyobject_ownref sixtyfour(PyLong_FromLong(64));
    pyobject_ownref hi_shifted(PyNumber_Lshift(hi.get(), sixtyfour));
    pyobject_ownref lo(PyLong_FromUnsignedLongLong(val.m_lo));
    return PyNumber_Or(hi_shifted.get(), lo.get());
  }

  PyObject *pyint_from_int(const dynd::dynd_int128 &val)
  {
    if (val.is_negative()) {
      if (val.m_hi == 0xffffffffffffffffULL &&
          (val.m_hi & 0x8000000000000000ULL) != 0) {
        return PyLong_FromLongLong(static_cast<int64_t>(val.m_lo));
      }
      pyobject_ownref absval(
          pyint_from_int(static_cast<dynd::dynd_uint128>(-val)));
      return PyNumber_Negative(absval.get());
    } else {
      return pyint_from_int(static_cast<dynd::dynd_uint128>(val));
    }
  }

  template <typename T>
  struct copy_int_kernel
      : base_kernel<copy_int_kernel<T>, kernel_request_host, 1> {
    void single(char *dst, char *const *src)
    {
      PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
      Py_XDECREF(*dst_obj);
      *dst_obj = NULL;
      *dst_obj = pyint_from_int(*reinterpret_cast<const T *>(src[0]));
    }
  };

  template <>
  struct copy_kernel<int8_type_id> : copy_int_kernel<int8_t> {
  };

  template <>
  struct copy_kernel<int16_type_id> : copy_int_kernel<int16_t> {
  };

  template <>
  struct copy_kernel<int32_type_id> : copy_int_kernel<int32_t> {
  };

  template <>
  struct copy_kernel<int64_type_id> : copy_int_kernel<int64_t> {
  };

  template <typename T>
  struct float_copy_kernel
      : base_kernel<float_copy_kernel<T>, kernel_request_host, 1> {
    void single(char *dst, char *const *src)
    {
      PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
      Py_XDECREF(*dst_obj);
      *dst_obj = NULL;
      *dst_obj = PyFloat_FromDouble(*reinterpret_cast<const T *>(src[0]));
    }
  };

  template <class T>
  struct complex_float_copy_kernel
      : base_kernel<complex_float_copy_kernel<T>, kernel_request_host, 1> {
    void single(char *dst, char *const *src)
    {
      PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
      Py_XDECREF(*dst_obj);
      *dst_obj = NULL;
      const complex<T> *val = reinterpret_cast<const complex<T> *>(src[0]);
      *dst_obj = PyComplex_FromDoubles(val->real(), val->imag());
    }
  };

  struct bytes_copy_kernel
      : base_kernel<bytes_copy_kernel, kernel_request_host, 1> {
    void single(char *dst, char *const *src)
    {
      PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
      Py_XDECREF(*dst_obj);
      *dst_obj = NULL;
      const bytes_type_data *bd =
          reinterpret_cast<const bytes_type_data *>(src[0]);
      *dst_obj = PyBytes_FromStringAndSize(bd->begin, bd->end - bd->begin);
    }
  };

  struct fixed_bytes_copy_kernel
      : base_kernel<fixed_bytes_copy_kernel, kernel_request_host, 1> {
    intptr_t data_size;

    fixed_bytes_copy_kernel(intptr_t data_size) : data_size(data_size) {}

    void single(char *dst, char *const *src)
    {
      PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
      Py_XDECREF(*dst_obj);
      *dst_obj = NULL;
      *dst_obj = PyBytes_FromStringAndSize(src[0], data_size);
    }

    static intptr_t instantiate(
        const arrfunc_type_data *DYND_UNUSED(self),
        const arrfunc_type *DYND_UNUSED(af_tp), char *DYND_UNUSED(data),
        void *ckb, intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
        const nd::array &DYND_UNUSED(kwds),
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      fixed_bytes_copy_kernel::create(
          ckb, kernreq, ckb_offset,
          src_tp[0].extended<fixedbytes_type>()->get_data_size());
      return ckb_offset;
    }
  };

  struct char_copy_kernel
      : base_kernel<char_copy_kernel, kernel_request_host, 1> {
    void single(char *dst, char *const *src)
    {
      PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
      Py_XDECREF(*dst_obj);
      *dst_obj = NULL;
      *dst_obj = PyUnicode_DecodeUTF32(src[0], 4, NULL, NULL);
    }
  };

  struct string_ascii_copy_kernel
      : base_kernel<string_ascii_copy_kernel, kernel_request_host, 1> {
    void single(char *dst, char *const *src)
    {
      PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
      Py_XDECREF(*dst_obj);
      *dst_obj = NULL;
      const string_type_data *sd =
          reinterpret_cast<const string_type_data *>(src[0]);
      *dst_obj = PyUnicode_DecodeASCII(sd->begin, sd->end - sd->begin, NULL);
    }
  };

  struct string_utf8_copy_kernel
      : base_kernel<string_utf8_copy_kernel, kernel_request_host, 1> {
    void single(char *dst, char *const *src)
    {
      PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
      Py_XDECREF(*dst_obj);
      *dst_obj = NULL;
      const string_type_data *sd =
          reinterpret_cast<const string_type_data *>(src[0]);
      *dst_obj = PyUnicode_DecodeUTF8(sd->begin, sd->end - sd->begin, NULL);
    }
  };

  struct string_utf16_copy_kernel
      : base_kernel<string_utf16_copy_kernel, kernel_request_host, 1> {
    void single(char *dst, char *const *src)
    {
      PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
      Py_XDECREF(*dst_obj);
      *dst_obj = NULL;
      const string_type_data *sd =
          reinterpret_cast<const string_type_data *>(src[0]);
      *dst_obj =
          PyUnicode_DecodeUTF16(sd->begin, sd->end - sd->begin, NULL, NULL);
    }
  };

  struct string_utf32_copy_kernel
      : base_kernel<string_utf32_copy_kernel, kernel_request_host, 1> {
    void single(char *dst, char *const *src)
    {
      PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
      Py_XDECREF(*dst_obj);
      *dst_obj = NULL;
      const string_type_data *sd =
          reinterpret_cast<const string_type_data *>(src[0]);
      *dst_obj =
          PyUnicode_DecodeUTF32(sd->begin, sd->end - sd->begin, NULL, NULL);
    }
  };

  struct string_copy_kernel : base_virtual_kernel<string_copy_kernel> {
    static intptr_t
    instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                char *data, void *ckb, intptr_t ckb_offset,
                const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                const ndt::type *src_tp, const char *const *src_arrmeta,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const nd::array &kwds,
                const std::map<nd::string, ndt::type> &tp_vars)
    {
      switch (src_tp[0].extended<base_string_type>()->get_encoding()) {
      case string_encoding_ascii:
        return string_ascii_copy_kernel::instantiate(
            self, self_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
            src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
      case string_encoding_utf_8:
        return string_utf8_copy_kernel::instantiate(
            self, self_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
            src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
      case string_encoding_ucs_2:
      case string_encoding_utf_16:
        return string_utf16_copy_kernel::instantiate(
            self, self_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
            src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
      case string_encoding_utf_32:
        return string_utf32_copy_kernel::instantiate(
            self, self_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
            src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
      default:
        break;
      }
    }
  };

  struct fixed_string_ascii_copy_kernel
      : base_kernel<fixed_string_ascii_copy_kernel, kernel_request_host, 1> {
    intptr_t data_size;

    fixed_string_ascii_copy_kernel(intptr_t data_size) : data_size(data_size) {}

    void single(char *dst, char *const *src)
    {
      PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
      Py_XDECREF(*dst_obj);
      *dst_obj = NULL;
      intptr_t size = std::find(src[0], src[0] + data_size, 0) - src[0];
      *dst_obj = PyUnicode_DecodeASCII(src[0], size, NULL);
    }

    static intptr_t instantiate(
        const arrfunc_type_data *DYND_UNUSED(self),
        const arrfunc_type *DYND_UNUSED(af_tp), char *DYND_UNUSED(data),
        void *ckb, intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
        const nd::array &DYND_UNUSED(kwds),
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      fixed_string_ascii_copy_kernel::create(ckb, kernreq, ckb_offset,
                                             src_tp[0].get_data_size());
      return ckb_offset;
    }
  };

  struct fixed_string_utf8_copy_kernel
      : base_kernel<fixed_string_utf8_copy_kernel, kernel_request_host, 1> {
    intptr_t data_size;

    fixed_string_utf8_copy_kernel(intptr_t data_size) : data_size(data_size) {}

    void single(char *dst, char *const *src)
    {
      PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
      Py_XDECREF(*dst_obj);
      *dst_obj = NULL;
      intptr_t size = std::find(src[0], src[0] + data_size, 0) - src[0];
      *dst_obj = PyUnicode_DecodeUTF8(src[0], size, NULL);
    }

    static intptr_t instantiate(
        const arrfunc_type_data *DYND_UNUSED(self),
        const arrfunc_type *DYND_UNUSED(af_tp), char *DYND_UNUSED(data),
        void *ckb, intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
        const nd::array &DYND_UNUSED(kwds),
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      fixed_string_utf8_copy_kernel::create(ckb, kernreq, ckb_offset,
                                            src_tp[0].get_data_size());
      return ckb_offset;
    }
  };

  struct fixed_string_utf16_copy_kernel
      : base_kernel<fixed_string_utf16_copy_kernel, kernel_request_host, 1> {
    intptr_t data_size;

    fixed_string_utf16_copy_kernel(intptr_t data_size) : data_size(data_size) {}

    void single(char *dst, char *const *src)
    {
      PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
      Py_XDECREF(*dst_obj);
      *dst_obj = NULL;
      const uint16_t *char16_src = reinterpret_cast<const uint16_t *>(src[0]);
      intptr_t size =
          std::find(char16_src, char16_src + (data_size >> 1), 0) - char16_src;
      *dst_obj = PyUnicode_DecodeUTF16(src[0], size * 2, NULL, NULL);
    }

    static intptr_t instantiate(
        const arrfunc_type_data *DYND_UNUSED(self),
        const arrfunc_type *DYND_UNUSED(af_tp), char *DYND_UNUSED(data),
        void *ckb, intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
        const nd::array &DYND_UNUSED(kwds),
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      fixed_string_utf16_copy_kernel::create(ckb, kernreq, ckb_offset,
                                             src_tp[0].get_data_size());
      return ckb_offset;
    }
  };

  struct fixed_string_utf32_copy_kernel
      : base_kernel<fixed_string_utf32_copy_kernel, kernel_request_host, 1> {
    intptr_t data_size;

    fixed_string_utf32_copy_kernel(intptr_t data_size) : data_size(data_size) {}

    void single(char *dst, char *const *src)
    {
      PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
      Py_XDECREF(*dst_obj);
      *dst_obj = NULL;
      const uint32_t *char32_src = reinterpret_cast<const uint32_t *>(src[0]);
      intptr_t size =
          std::find(char32_src, char32_src + (data_size >> 2), 0) - char32_src;
      *dst_obj = PyUnicode_DecodeUTF32(src[0], size * 4, NULL, NULL);
    }

    static intptr_t instantiate(
        const arrfunc_type_data *DYND_UNUSED(self),
        const arrfunc_type *DYND_UNUSED(af_tp), char *DYND_UNUSED(data),
        void *ckb, intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const char *const *DYND_UNUSED(src_arrmeta),
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
        const nd::array &DYND_UNUSED(kwds),
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      fixed_string_utf32_copy_kernel::create(ckb, kernreq, ckb_offset,
                                             src_tp[0].get_data_size());
      return ckb_offset;
    }
  };

  struct fixed_string_copy_kernel
      : base_virtual_kernel<fixed_string_copy_kernel> {
    static intptr_t
    instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                char *data, void *ckb, intptr_t ckb_offset,
                const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                const ndt::type *src_tp, const char *const *src_arrmeta,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const nd::array &kwds,
                const std::map<nd::string, ndt::type> &tp_vars)
    {
      switch (src_tp[0].extended<base_string_type>()->get_encoding()) {
      case string_encoding_ascii:
        return fixed_string_ascii_copy_kernel::instantiate(
            self, self_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
            src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
      case string_encoding_utf_8:
        return fixed_string_utf8_copy_kernel::instantiate(
            self, self_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
            src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
      case string_encoding_ucs_2:
      case string_encoding_utf_16:
        return fixed_string_utf16_copy_kernel::instantiate(
            self, self_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
            src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
      case string_encoding_utf_32:
        return fixed_string_utf32_copy_kernel::instantiate(
            self, self_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
            src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
      default:
        break;
      }
    }
  };

  struct date_copy_kernel
      : base_kernel<date_copy_kernel, kernel_request_host, 1> {
    ndt::type src_tp;
    const char *src_arrmeta;

    date_copy_kernel(ndt::type src_tp, const char *src_arrmeta)
        : src_tp(src_tp), src_arrmeta(src_arrmeta)
    {
    }

    void single(char *dst, char *const *src)
    {
      PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
      Py_XDECREF(*dst_obj);
      *dst_obj = NULL;
      const date_type *dd = src_tp.extended<date_type>();
      date_ymd ymd = dd->get_ymd(src_arrmeta, src[0]);
      *dst_obj = PyDate_FromDate(ymd.year, ymd.month, ymd.day);
    }

    static intptr_t instantiate(
        const arrfunc_type_data *DYND_UNUSED(self),
        const arrfunc_type *DYND_UNUSED(af_tp), char *DYND_UNUSED(data),
        void *ckb, intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const char *const *src_arrmeta,
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
        const nd::array &DYND_UNUSED(kwds),
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      date_copy_kernel::create(ckb, kernreq, ckb_offset, src_tp[0],
                               src_arrmeta[0]);
      return ckb_offset;
    }
  };

  struct time_copy_kernel
      : base_kernel<time_copy_kernel, kernel_request_host, 1> {
    ndt::type src_tp;
    const char *src_arrmeta;

    time_copy_kernel(ndt::type src_tp, const char *src_arrmeta)
        : src_tp(src_tp), src_arrmeta(src_arrmeta)
    {
    }

    void single(char *dst, char *const *src)
    {
      PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
      Py_XDECREF(*dst_obj);
      *dst_obj = NULL;
      const time_type *tt = src_tp.extended<time_type>();
      time_hmst hmst = tt->get_time(src_arrmeta, src[0]);
      *dst_obj = PyTime_FromTime(hmst.hour, hmst.minute, hmst.second,
                                 hmst.tick / DYND_TICKS_PER_MICROSECOND);
    }

    static intptr_t instantiate(
        const arrfunc_type_data *DYND_UNUSED(self),
        const arrfunc_type *DYND_UNUSED(af_tp), char *DYND_UNUSED(data),
        void *ckb, intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const char *const *src_arrmeta,
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
        const nd::array &DYND_UNUSED(kwds),
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      time_copy_kernel::create(ckb, kernreq, ckb_offset, src_tp[0],
                               src_arrmeta[0]);
      return ckb_offset;
    }
  };

  struct datetime_copy_kernel
      : base_kernel<datetime_copy_kernel, kernel_request_host, 1> {
    ndt::type src_tp;
    const char *src_arrmeta;

    datetime_copy_kernel(ndt::type src_tp, const char *src_arrmeta)
        : src_tp(src_tp), src_arrmeta(src_arrmeta)
    {
    }

    void single(char *dst, char *const *src)
    {
      PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
      Py_XDECREF(*dst_obj);
      *dst_obj = NULL;
      const datetime_type *dd = src_tp.extended<datetime_type>();
      int32_t year, month, day, hour, minute, second, tick;
      dd->get_cal(src_arrmeta, src[0], year, month, day, hour, minute, second,
                  tick);
      int32_t usecond = tick / 10;
      *dst_obj = PyDateTime_FromDateAndTime(year, month, day, hour, minute,
                                            second, usecond);
    }

    static intptr_t instantiate(
        const arrfunc_type_data *DYND_UNUSED(self),
        const arrfunc_type *DYND_UNUSED(af_tp), char *DYND_UNUSED(data),
        void *ckb, intptr_t ckb_offset, const ndt::type &DYND_UNUSED(dst_tp),
        const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc),
        const ndt::type *src_tp, const char *const *src_arrmeta,
        kernel_request_t kernreq, const eval::eval_context *DYND_UNUSED(ectx),
        const nd::array &DYND_UNUSED(kwds),
        const std::map<nd::string, ndt::type> &DYND_UNUSED(tp_vars))
    {
      datetime_copy_kernel::create(ckb, kernreq, ckb_offset, src_tp[0],
                                   src_arrmeta[0]);
      return ckb_offset;
    }
  };

  struct type_copy_kernel
      : base_kernel<type_copy_kernel, kernel_request_host, 1> {
    void single(char *dst, char *const *src)
    {
      PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
      Py_XDECREF(*dst_obj);
      *dst_obj = NULL;
      ndt::type tp(reinterpret_cast<const type_type_data *>(src[0])->tp, true);
      *dst_obj = wrap_ndt_type(std::move(tp));
    }
  };

  struct pointer_copy_kernel
      : base_kernel<pointer_copy_kernel, kernel_request_host, 1> {
    void single(char *dst, char *const *src)
    {
      PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
      Py_XDECREF(*dst_obj);
      *dst_obj = NULL;
      ckernel_prefix *copy_value = get_child_ckernel();
      expr_single_t copy_value_fn = copy_value->get_function<expr_single_t>();
      // The src value is a pointer, and copy_value_fn expects a pointer
      // to that pointer
      char **src_ptr = reinterpret_cast<char **>(src[0]);
      copy_value_fn(dst, src_ptr, copy_value);
    }

    void destruct_children() { get_child_ckernel()->destroy(); }
  };

} // namespace pydynd::nd
} // namespace pydynd
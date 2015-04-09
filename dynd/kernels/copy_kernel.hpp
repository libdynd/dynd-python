#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/base_virtual_kernel.hpp>

#include "config.hpp"

namespace pydynd {
namespace nd {

  struct copy_to_pyobject_virtual_kernel
      : base_virtual_kernel<copy_to_pyobject_virtual_kernel> {
    static intptr_t
    instantiate(const arrfunc_type_data *self_af, const arrfunc_type *af_tp,
                char *data, void *ckb, intptr_t ckb_offset,
                const dynd::ndt::type &dst_tp, const char *dst_arrmeta,
                intptr_t nsrc, const dynd::ndt::type *src_tp,
                const char *const *src_arrmeta, kernel_request_t kernreq,
                const eval::eval_context *ectx, const dynd::nd::array &kwds,
                const std::map<dynd::nd::string, dynd::ndt::type> &tp_vars);
  };

  template <type_id_t src_type_id>
  struct copy_to_pyobject_kernel;

  template <>
  struct copy_to_pyobject_kernel<bool_type_id>
      : base_kernel<copy_to_pyobject_kernel<bool_type_id>, kernel_request_host,
                    1> {
    void single(char *dst, char *const *src)
    {
      PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
      Py_XDECREF(*dst_obj);
      *dst_obj = (*src[0] != 0) ? Py_True : Py_False;
      Py_INCREF(*dst_obj);
    }

    static ndt::type make_type() { return ndt::type("(bool) -> void"); }
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
  struct copy_to_pyobject_kernel<int8_type_id> : copy_int_kernel<int8_t> {
  };

  template <>
  struct copy_to_pyobject_kernel<int16_type_id> : copy_int_kernel<int16_t> {
  };

  template <>
  struct copy_to_pyobject_kernel<int32_type_id> : copy_int_kernel<int32_t> {
  };

  template <>
  struct copy_to_pyobject_kernel<int64_type_id> : copy_int_kernel<int64_t> {
  };

  template <>
  struct copy_to_pyobject_kernel<int128_type_id>
      : copy_int_kernel<dynd_int128> {
  };

  template <>
  struct copy_to_pyobject_kernel<uint8_type_id> : copy_int_kernel<uint8_t> {
  };

  template <>
  struct copy_to_pyobject_kernel<uint16_type_id> : copy_int_kernel<uint16_t> {
  };

  template <>
  struct copy_to_pyobject_kernel<uint32_type_id> : copy_int_kernel<uint32_t> {
  };

  template <>
  struct copy_to_pyobject_kernel<uint64_type_id> : copy_int_kernel<uint64_t> {
  };

  template <>
  struct copy_to_pyobject_kernel<uint128_type_id>
      : copy_int_kernel<dynd_uint128> {
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

  template <>
  struct copy_to_pyobject_kernel<float16_type_id>
      : float_copy_kernel<dynd_float16> {
  };

  template <>
  struct copy_to_pyobject_kernel<float32_type_id> : float_copy_kernel<float> {
  };

  template <>
  struct copy_to_pyobject_kernel<float64_type_id> : float_copy_kernel<double> {
  };

  template <class T>
  struct complex_float_copy_kernel
      : base_kernel<complex_float_copy_kernel<T>, kernel_request_host, 1> {
    void single(char *dst, char *const *src)
    {
      PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
      Py_XDECREF(*dst_obj);
      *dst_obj = NULL;
      const dynd::complex<T> *val =
          reinterpret_cast<const dynd::complex<T> *>(src[0]);
      *dst_obj = PyComplex_FromDoubles(val->real(), val->imag());
    }
  };

  template <>
  struct copy_to_pyobject_kernel<complex_float32_type_id>
      : complex_float_copy_kernel<float> {
  };

  template <>
  struct copy_to_pyobject_kernel<complex_float64_type_id>
      : complex_float_copy_kernel<double> {
  };

  template <>
  struct copy_to_pyobject_kernel<bytes_type_id>
      : base_kernel<copy_to_pyobject_kernel<bytes_type_id>, kernel_request_host,
                    1> {
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

  template <>
  struct copy_to_pyobject_kernel<fixedbytes_type_id>
      : base_kernel<copy_to_pyobject_kernel<fixedbytes_type_id>,
                    kernel_request_host, 1> {
    intptr_t data_size;

    copy_to_pyobject_kernel(intptr_t data_size) : data_size(data_size) {}

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
      copy_to_pyobject_kernel::make(
          ckb, kernreq, ckb_offset,
          src_tp[0].extended<fixedbytes_type>()->get_data_size());
      return ckb_offset;
    }
  };

  template <>
  struct copy_to_pyobject_kernel<char_type_id>
      : base_kernel<copy_to_pyobject_kernel<char_type_id>, kernel_request_host,
                    1> {
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

  template <>
  struct copy_to_pyobject_kernel<string_type_id>
      : base_virtual_kernel<copy_to_pyobject_kernel<string_type_id>> {
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
        throw std::runtime_error("no string_copy_kernel");
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
      fixed_string_ascii_copy_kernel::make(ckb, kernreq, ckb_offset,
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
      fixed_string_utf8_copy_kernel::make(ckb, kernreq, ckb_offset,
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
      fixed_string_utf16_copy_kernel::make(ckb, kernreq, ckb_offset,
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
      fixed_string_utf32_copy_kernel::make(ckb, kernreq, ckb_offset,
                                           src_tp[0].get_data_size());
      return ckb_offset;
    }
  };

  template <>
  struct copy_to_pyobject_kernel<fixedstring_type_id>
      : base_virtual_kernel<copy_to_pyobject_kernel<fixedstring_type_id>> {
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
        throw std::runtime_error("no fixed_string_copy_kernel");
      }
    }
  };

  template <>
  struct copy_to_pyobject_kernel<date_type_id>
      : base_kernel<copy_to_pyobject_kernel<date_type_id>, kernel_request_host,
                    1> {
    ndt::type src_tp;
    const char *src_arrmeta;

    copy_to_pyobject_kernel(ndt::type src_tp, const char *src_arrmeta)
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
      copy_to_pyobject_kernel::make(ckb, kernreq, ckb_offset, src_tp[0],
                                    src_arrmeta[0]);
      return ckb_offset;
    }
  };

  template <>
  struct copy_to_pyobject_kernel<time_type_id>
      : base_kernel<copy_to_pyobject_kernel<time_type_id>, kernel_request_host,
                    1> {
    ndt::type src_tp;
    const char *src_arrmeta;

    copy_to_pyobject_kernel(ndt::type src_tp, const char *src_arrmeta)
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
      copy_to_pyobject_kernel::make(ckb, kernreq, ckb_offset, src_tp[0],
                                    src_arrmeta[0]);
      return ckb_offset;
    }
  };

  template <>
  struct copy_to_pyobject_kernel<datetime_type_id>
      : base_kernel<copy_to_pyobject_kernel<datetime_type_id>,
                    kernel_request_host, 1> {
    ndt::type src_tp;
    const char *src_arrmeta;

    copy_to_pyobject_kernel(ndt::type src_tp, const char *src_arrmeta)
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
      copy_to_pyobject_kernel::make(ckb, kernreq, ckb_offset, src_tp[0],
                                    src_arrmeta[0]);
      return ckb_offset;
    }
  };

  template <>
  struct copy_to_pyobject_kernel<type_type_id>
      : base_kernel<copy_to_pyobject_kernel<type_type_id>, kernel_request_host,
                    1> {
    void single(char *dst, char *const *src)
    {
      PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
      Py_XDECREF(*dst_obj);
      *dst_obj = NULL;
      ndt::type tp(reinterpret_cast<const type_type_data *>(src[0])->tp, true);
      *dst_obj = wrap_ndt_type(std::move(tp));
    }
  };

  // TODO: Should make a more efficient strided kernel function
  template <>
  struct copy_to_pyobject_kernel<option_type_id>
      : base_kernel<copy_to_pyobject_kernel<option_type_id>,
                    kernel_request_host, 1> {
    intptr_t m_copy_value_offset;

    void single(char *dst, char *const *src)
    {
      PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
      Py_XDECREF(*dst_obj);
      *dst_obj = NULL;
      dynd::ckernel_prefix *is_avail = get_child_ckernel();
      dynd::expr_single_t is_avail_fn =
          is_avail->get_function<dynd::expr_single_t>();
      dynd::ckernel_prefix *copy_value = get_child_ckernel(m_copy_value_offset);
      dynd::expr_single_t copy_value_fn =
          copy_value->get_function<dynd::expr_single_t>();
      char value_is_avail = 0;
      is_avail_fn(&value_is_avail, src, is_avail);
      if (value_is_avail != 0) {
        copy_value_fn(dst, src, copy_value);
      } else {
        *dst_obj = Py_None;
        Py_INCREF(*dst_obj);
      }
    }

    void destruct_children()
    {
      get_child_ckernel()->destroy();
      base.destroy_child_ckernel(m_copy_value_offset);
    }

    static intptr_t
    instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                char *data, void *ckb, intptr_t ckb_offset,
                const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                const ndt::type *src_tp, const char *const *src_arrmeta,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const nd::array &kwds,
                const std::map<nd::string, ndt::type> &tp_vars)
    {
      intptr_t root_ckb_offset = ckb_offset;
      copy_to_pyobject_kernel *self_ck =
          copy_to_pyobject_kernel::make(ckb, kernreq, ckb_offset);
      const arrfunc_type_data *is_avail_af =
          src_tp[0].extended<option_type>()->get_is_avail_arrfunc();
      const arrfunc_type *is_avail_af_tp =
          src_tp[0].extended<option_type>()->get_is_avail_arrfunc_type();
      ckb_offset = is_avail_af->instantiate(
          is_avail_af, is_avail_af_tp, NULL, ckb, ckb_offset,
          dynd::ndt::make_type<dynd_bool>(), NULL, nsrc, src_tp, src_arrmeta,
          kernel_request_single, ectx, dynd::nd::array(), tp_vars);
      reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
          ->reserve(ckb_offset);
      self_ck = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
                    ->get_at<copy_to_pyobject_kernel>(root_ckb_offset);
      self_ck->m_copy_value_offset = ckb_offset - root_ckb_offset;
      dynd::ndt::type src_value_tp =
          src_tp[0].extended<option_type>()->get_value_type();
      ckb_offset = self->instantiate(self, self_tp, NULL, ckb, ckb_offset,
                                     dst_tp, dst_arrmeta, nsrc, &src_value_tp,
                                     src_arrmeta, kernel_request_single, ectx,
                                     dynd::nd::array(), tp_vars);
      return ckb_offset;
    }
  };

  template <>
  struct copy_to_pyobject_kernel<fixed_dim_type_id>
      : base_kernel<copy_to_pyobject_kernel<fixed_dim_type_id>,
                    kernel_request_host, 1> {
    intptr_t dim_size, stride;

    copy_to_pyobject_kernel(intptr_t dim_size, intptr_t stride)
        : dim_size(dim_size), stride(stride)
    {
    }

    void single(char *dst, char *const *src)
    {
      PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
      Py_XDECREF(*dst_obj);
      *dst_obj = NULL;
      pyobject_ownref lst(PyList_New(dim_size));
      ckernel_prefix *copy_el = get_child_ckernel();
      expr_strided_t copy_el_fn = copy_el->get_function<expr_strided_t>();
      copy_el_fn(reinterpret_cast<char *>(((PyListObject *)lst.get())->ob_item),
                 sizeof(PyObject *), src, &stride, dim_size, copy_el);
      if (PyErr_Occurred()) {
        throw std::exception();
      }
      *dst_obj = lst.release();
    }

    void destruct_children() { get_child_ckernel()->destroy(); }

    static intptr_t
    instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                char *data, void *ckb, intptr_t ckb_offset,
                const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                const ndt::type *src_tp, const char *const *src_arrmeta,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const nd::array &kwds,
                const std::map<nd::string, ndt::type> &tp_vars)
    {
      intptr_t dim_size, stride;
      ndt::type el_tp;
      const char *el_arrmeta;
      if (src_tp[0].get_as_strided(src_arrmeta[0], &dim_size, &stride, &el_tp,
                                   &el_arrmeta)) {
        copy_to_pyobject_kernel::make(ckb, kernreq, ckb_offset, dim_size,
                                      stride);
        return copy_to_pyobject_virtual_kernel::instantiate(
            self, self_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
            &el_tp, &el_arrmeta, kernel_request_strided, ectx, kwds, tp_vars);
      }

      throw std::runtime_error("cannot process as strided");
    }
  };

  template <>
  struct copy_to_pyobject_kernel<var_dim_type_id>
      : base_kernel<copy_to_pyobject_kernel<var_dim_type_id>,
                    kernel_request_host, 1> {
    intptr_t offset, stride;

    copy_to_pyobject_kernel(intptr_t offset, intptr_t stride)
        : offset(offset), stride(stride)
    {
    }

    void single(char *dst, char *const *src)
    {
      PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
      Py_XDECREF(*dst_obj);
      *dst_obj = NULL;
      const var_dim_type_data *vd =
          reinterpret_cast<const var_dim_type_data *>(src[0]);
      pyobject_ownref lst(PyList_New(vd->size));
      dynd::ckernel_prefix *copy_el = get_child_ckernel();
      expr_strided_t copy_el_fn = copy_el->get_function<expr_strided_t>();
      char *el_src = vd->begin + offset;
      copy_el_fn(reinterpret_cast<char *>(((PyListObject *)lst.get())->ob_item),
                 sizeof(PyObject *), &el_src, &stride, vd->size, copy_el);
      if (PyErr_Occurred()) {
        throw std::exception();
      }
      *dst_obj = lst.release();
    }

    void destruct_children() { get_child_ckernel()->destroy(); }

    static intptr_t
    instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                char *data, void *ckb, intptr_t ckb_offset,
                const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                const ndt::type *src_tp, const char *const *src_arrmeta,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const nd::array &kwds,
                const std::map<nd::string, ndt::type> &tp_vars)
    {
      copy_to_pyobject_kernel::make(
          ckb, kernreq, ckb_offset,
          reinterpret_cast<const var_dim_type_arrmeta *>(src_arrmeta[0])
              ->offset,
          reinterpret_cast<const var_dim_type_arrmeta *>(src_arrmeta[0])
              ->stride);
      ndt::type el_tp = src_tp[0].extended<var_dim_type>()->get_element_type();
      const char *el_arrmeta = src_arrmeta[0] + sizeof(var_dim_type_arrmeta);
      return self->instantiate(self, self_tp, data, ckb, ckb_offset, dst_tp,
                               dst_arrmeta, nsrc, &el_tp, &el_arrmeta,
                               kernel_request_strided, ectx, kwds, tp_vars);
    }
  };

  // TODO: Should make a more efficient strided kernel function
  template <>
  struct copy_to_pyobject_kernel<struct_type_id>
      : base_kernel<copy_to_pyobject_kernel<struct_type_id>,
                    kernel_request_host, 1> {
    dynd::ndt::type m_src_tp;
    const char *m_src_arrmeta;
    std::vector<intptr_t> m_copy_el_offsets;
    pyobject_ownref m_field_names;

    void single(char *dst, char *const *src)
    {
      PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
      Py_XDECREF(*dst_obj);
      *dst_obj = NULL;
      intptr_t field_count =
          m_src_tp.extended<base_tuple_type>()->get_field_count();
      const uintptr_t *field_offsets =
          m_src_tp.extended<base_tuple_type>()->get_data_offsets(m_src_arrmeta);
      pyobject_ownref dct(PyDict_New());
      for (intptr_t i = 0; i < field_count; ++i) {
        dynd::ckernel_prefix *copy_el = get_child_ckernel(m_copy_el_offsets[i]);
        dynd::expr_single_t copy_el_fn =
            copy_el->get_function<dynd::expr_single_t>();
        char *el_src = src[0] + field_offsets[i];
        pyobject_ownref el;
        copy_el_fn(reinterpret_cast<char *>(el.obj_addr()), &el_src, copy_el);
        PyDict_SetItem(dct.get(), PyTuple_GET_ITEM(m_field_names.get(), i),
                       el.get());
      }
      if (PyErr_Occurred()) {
        throw std::exception();
      }
      *dst_obj = dct.release();
    }

    void destruct_children()
    {
      for (size_t i = 0; i < m_copy_el_offsets.size(); ++i) {
        base.destroy_child_ckernel(m_copy_el_offsets[i]);
      }
    }

    static intptr_t
    instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                char *data, void *ckb, intptr_t ckb_offset,
                const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                const ndt::type *src_tp, const char *const *src_arrmeta,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const nd::array &kwds,
                const std::map<nd::string, ndt::type> &tp_vars)
    {
      intptr_t root_ckb_offset = ckb_offset;
      copy_to_pyobject_kernel *self_ck =
          copy_to_pyobject_kernel::make(ckb, kernreq, ckb_offset);
      self_ck->m_src_tp = src_tp[0];
      self_ck->m_src_arrmeta = src_arrmeta[0];
      intptr_t field_count =
          src_tp[0].extended<base_struct_type>()->get_field_count();
      const dynd::ndt::type *field_types =
          src_tp[0].extended<base_struct_type>()->get_field_types_raw();
      const uintptr_t *arrmeta_offsets =
          src_tp[0].extended<base_struct_type>()->get_arrmeta_offsets_raw();
      self_ck->m_field_names.reset(PyTuple_New(field_count));
      for (intptr_t i = 0; i < field_count; ++i) {
        const string_type_data &rawname =
            src_tp[0].extended<base_struct_type>()->get_field_name_raw(i);
        pyobject_ownref name(PyUnicode_DecodeUTF8(
            rawname.begin, rawname.end - rawname.begin, NULL));
        PyTuple_SET_ITEM(self_ck->m_field_names.get(), i, name.release());
      }
      self_ck->m_copy_el_offsets.resize(field_count);
      for (intptr_t i = 0; i < field_count; ++i) {
        reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
            ->reserve(ckb_offset);
        self_ck = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
                      ->get_at<copy_to_pyobject_kernel>(root_ckb_offset);
        self_ck->m_copy_el_offsets[i] = ckb_offset - root_ckb_offset;
        const char *field_arrmeta = src_arrmeta[0] + arrmeta_offsets[i];
        ckb_offset = self->instantiate(
            self, self_tp, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
            &field_types[i], &field_arrmeta, kernel_request_single, ectx,
            dynd::nd::array(), tp_vars);
      }
      return ckb_offset;
    }
  };

  // TODO: Should make a more efficient strided kernel function
  template <>
  struct copy_to_pyobject_kernel<tuple_type_id>
      : base_kernel<copy_to_pyobject_kernel<tuple_type_id>, kernel_request_host,
                    1> {
    ndt::type src_tp;
    const char *src_arrmeta;
    std::vector<intptr_t> m_copy_el_offsets;

    copy_to_pyobject_kernel(ndt::type src_tp, const char *src_arrmeta)
        : src_tp(src_tp), src_arrmeta(src_arrmeta)
    {
    }

    void single(char *dst, char *const *src)
    {
      PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
      Py_XDECREF(*dst_obj);
      *dst_obj = NULL;
      intptr_t field_count =
          src_tp.extended<base_tuple_type>()->get_field_count();
      const uintptr_t *field_offsets =
          src_tp.extended<base_tuple_type>()->get_data_offsets(src_arrmeta);
      pyobject_ownref tup(PyTuple_New(field_count));
      for (intptr_t i = 0; i < field_count; ++i) {
        ckernel_prefix *copy_el = get_child_ckernel(m_copy_el_offsets[i]);
        expr_single_t copy_el_fn = copy_el->get_function<expr_single_t>();
        char *el_src = src[0] + field_offsets[i];
        char *el_dst =
            reinterpret_cast<char *>(((PyTupleObject *)tup.get())->ob_item + i);
        copy_el_fn(el_dst, &el_src, copy_el);
      }
      if (PyErr_Occurred()) {
        throw std::exception();
      }
      *dst_obj = tup.release();
    }

    void destruct_children()
    {
      for (size_t i = 0; i < m_copy_el_offsets.size(); ++i) {
        base.destroy_child_ckernel(m_copy_el_offsets[i]);
      }
    }

    static intptr_t
    instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                char *data, void *ckb, intptr_t ckb_offset,
                const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                const ndt::type *src_tp, const char *const *src_arrmeta,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const nd::array &kwds,
                const std::map<nd::string, ndt::type> &tp_vars)
    {
      intptr_t root_ckb_offset = ckb_offset;
      copy_to_pyobject_kernel *self_ck = copy_to_pyobject_kernel::make(
          ckb, kernreq, ckb_offset, src_tp[0], src_arrmeta[0]);
      intptr_t field_count =
          src_tp[0].extended<base_tuple_type>()->get_field_count();
      const ndt::type *field_types =
          src_tp[0].extended<base_tuple_type>()->get_field_types_raw();
      const uintptr_t *arrmeta_offsets =
          src_tp[0].extended<base_tuple_type>()->get_arrmeta_offsets_raw();
      self_ck->m_copy_el_offsets.resize(field_count);
      for (intptr_t i = 0; i < field_count; ++i) {
        reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
            ->reserve(ckb_offset);
        self_ck = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
                      ->get_at<copy_to_pyobject_kernel>(root_ckb_offset);
        self_ck->m_copy_el_offsets[i] = ckb_offset - root_ckb_offset;
        const char *field_arrmeta = src_arrmeta[0] + arrmeta_offsets[i];
        ckb_offset = self->instantiate(
            self, self_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
            &field_types[i], &field_arrmeta, kernel_request_single, ectx, kwds,
            tp_vars);
      }
      return ckb_offset;
    }
  };

  template <>
  struct copy_to_pyobject_kernel<pointer_type_id>
      : base_kernel<copy_to_pyobject_kernel<pointer_type_id>,
                    kernel_request_host, 1> {
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

    static intptr_t
    instantiate(const arrfunc_type_data *self, const arrfunc_type *self_tp,
                char *data, void *ckb, intptr_t ckb_offset,
                const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                const ndt::type *src_tp, const char *const *src_arrmeta,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const nd::array &kwds,
                const std::map<nd::string, ndt::type> &tp_vars)
    {
      copy_to_pyobject_kernel::make(ckb, kernreq, ckb_offset);
      dynd::ndt::type src_value_tp =
          src_tp[0].extended<pointer_type>()->get_target_type();
      return self->instantiate(self, self_tp, NULL, ckb, ckb_offset, dst_tp,
                               dst_arrmeta, nsrc, &src_value_tp, src_arrmeta,
                               kernel_request_single, ectx, dynd::nd::array(),
                               tp_vars);
    }
  };

} // namespace pydynd::nd
} // namespace pydynd
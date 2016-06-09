//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/types/fixed_bytes_type.hpp>

using namespace dynd;

template <typename Arg0Type, typename Enable = void>
struct assign_to_pyobject_kernel;

template <>
struct assign_to_pyobject_kernel<bool> : nd::base_strided_kernel<assign_to_pyobject_kernel<bool>, 1> {
  void single(char *dst, char *const *src)
  {
    Py_XDECREF(*reinterpret_cast<PyObject **>(dst));
    *reinterpret_cast<PyObject **>(dst) = *reinterpret_cast<bool *>(src[0]) ? Py_True : Py_False;
    Py_INCREF(*reinterpret_cast<PyObject **>(dst));
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

PyObject *pyint_from_int(uint64_t v) { return PyLong_FromUnsignedLongLong(v); }
#endif

PyObject *pyint_from_int(const dynd::uint128 &val)
{
  if (val.m_hi == 0ULL) {
    return PyLong_FromUnsignedLongLong(val.m_lo);
  }
  // Use the pynumber methods to shift and or together the 64 bit parts
  pydynd::py_ref hi = pydynd::capture_if_not_null(PyLong_FromUnsignedLongLong(val.m_hi));
  pydynd::py_ref sixtyfour = pydynd::capture_if_not_null(PyLong_FromLong(64));
  pydynd::py_ref hi_shifted = pydynd::capture_if_not_null(PyNumber_Lshift(hi.get(), sixtyfour.get()));
  pydynd::py_ref lo = pydynd::capture_if_not_null(PyLong_FromUnsignedLongLong(val.m_lo));
  return PyNumber_Or(hi_shifted.get(), lo.get());
}

PyObject *pyint_from_int(const dynd::int128 &val)
{
  if (val.is_negative()) {
    if (val.m_hi == 0xffffffffffffffffULL && (val.m_hi & 0x8000000000000000ULL) != 0) {
      return PyLong_FromLongLong(static_cast<int64_t>(val.m_lo));
    }
    pydynd::py_ref absval = pydynd::capture_if_not_null(pyint_from_int(static_cast<dynd::uint128>(-val)));
    return PyNumber_Negative(absval.get());
  }
  else {
    return pyint_from_int(static_cast<dynd::uint128>(val));
  }
}

template <typename T>
struct assign_int_kernel : dynd::nd::base_strided_kernel<assign_int_kernel<T>, 1> {
  void single(char *dst, char *const *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    *dst_obj = pyint_from_int(*reinterpret_cast<const T *>(src[0]));
  }
};

template <>
struct assign_to_pyobject_kernel<int8_t> : assign_int_kernel<int8_t> {
};

template <>
struct assign_to_pyobject_kernel<int16_t> : assign_int_kernel<int16_t> {
};

template <>
struct assign_to_pyobject_kernel<int32_t> : assign_int_kernel<int32_t> {
};

template <>
struct assign_to_pyobject_kernel<int64_t> : assign_int_kernel<int64_t> {
};

template <>
struct assign_to_pyobject_kernel<int128> : assign_int_kernel<dynd::int128> {
};

template <>
struct assign_to_pyobject_kernel<uint8_t> : assign_int_kernel<uint8_t> {
};

template <>
struct assign_to_pyobject_kernel<uint16_t> : assign_int_kernel<uint16_t> {
};

template <>
struct assign_to_pyobject_kernel<uint32_t> : assign_int_kernel<uint32_t> {
};

template <>
struct assign_to_pyobject_kernel<uint64_t> : assign_int_kernel<uint64_t> {
};

template <>
struct assign_to_pyobject_kernel<uint128> : assign_int_kernel<dynd::uint128> {
};

template <typename T>
struct float_assign_kernel : dynd::nd::base_strided_kernel<float_assign_kernel<T>, 1> {
  void single(char *dst, char *const *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    *dst_obj = PyFloat_FromDouble(*reinterpret_cast<const T *>(src[0]));
  }
};

template <>
struct assign_to_pyobject_kernel<float> : float_assign_kernel<float> {
};

template <>
struct assign_to_pyobject_kernel<double> : float_assign_kernel<double> {
};

template <class T>
struct complex_float_assign_kernel : dynd::nd::base_strided_kernel<complex_float_assign_kernel<T>, 1> {
  void single(char *dst, char *const *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    const dynd::complex<T> *val = reinterpret_cast<const dynd::complex<T> *>(src[0]);
    *dst_obj = PyComplex_FromDoubles(val->real(), val->imag());
  }
};

template <>
struct assign_to_pyobject_kernel<dynd::complex<float>> : complex_float_assign_kernel<float> {
};

template <>
struct assign_to_pyobject_kernel<dynd::complex<double>> : complex_float_assign_kernel<double> {
};

template <>
struct assign_to_pyobject_kernel<dynd::bytes>
    : dynd::nd::base_strided_kernel<assign_to_pyobject_kernel<dynd::bytes>, 1> {
  void single(char *dst, char *const *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    const dynd::bytes *bd = reinterpret_cast<const dynd::bytes *>(src[0]);
    *dst_obj = PyBytes_FromStringAndSize(bd->begin(), bd->end() - bd->begin());
  }
};

template <>
struct assign_to_pyobject_kernel<ndt::fixed_bytes_type>
    : dynd::nd::base_strided_kernel<assign_to_pyobject_kernel<ndt::fixed_bytes_type>, 1> {
  intptr_t data_size;

  assign_to_pyobject_kernel(intptr_t data_size) : data_size(data_size) {}

  void single(char *dst, char *const *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    *dst_obj = PyBytes_FromStringAndSize(src[0], data_size);
  }
};

template <>
struct assign_to_pyobject_kernel<char> : dynd::nd::base_strided_kernel<assign_to_pyobject_kernel<char>, 1> {
  void single(char *dst, char *const *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    *dst_obj = PyUnicode_DecodeUTF32(src[0], 4, NULL, NULL);
  }
};

struct string_ascii_assign_kernel : dynd::nd::base_strided_kernel<string_ascii_assign_kernel, 1> {
  void single(char *dst, char *const *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    const dynd::string *sd = reinterpret_cast<const dynd::string *>(src[0]);
    *dst_obj = PyUnicode_DecodeASCII(sd->begin(), sd->end() - sd->begin(), NULL);
  }
};

struct string_utf8_assign_kernel : dynd::nd::base_strided_kernel<string_utf8_assign_kernel, 1> {
  void single(char *dst, char *const *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    const dynd::string *sd = reinterpret_cast<const dynd::string *>(src[0]);
    *dst_obj = PyUnicode_DecodeUTF8(sd->begin(), sd->end() - sd->begin(), NULL);
  }
};

struct string_utf16_assign_kernel : dynd::nd::base_strided_kernel<string_utf16_assign_kernel, 1> {
  void single(char *dst, char *const *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    const dynd::string *sd = reinterpret_cast<const dynd::string *>(src[0]);
    *dst_obj = PyUnicode_DecodeUTF16(sd->begin(), sd->end() - sd->begin(), NULL, NULL);
  }
};

struct string_utf32_assign_kernel : dynd::nd::base_strided_kernel<string_utf32_assign_kernel, 1> {
  void single(char *dst, char *const *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    const dynd::string *sd = reinterpret_cast<const dynd::string *>(src[0]);
    *dst_obj = PyUnicode_DecodeUTF32(sd->begin(), sd->end() - sd->begin(), NULL, NULL);
  }
};

struct fixed_string_ascii_assign_kernel : dynd::nd::base_strided_kernel<fixed_string_ascii_assign_kernel, 1> {
  intptr_t data_size;

  fixed_string_ascii_assign_kernel(intptr_t data_size) : data_size(data_size) {}

  void single(char *dst, char *const *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    intptr_t size = std::find(src[0], src[0] + data_size, 0) - src[0];
    *dst_obj = PyUnicode_DecodeASCII(src[0], size, NULL);
  }
};

struct fixed_string_utf8_assign_kernel : dynd::nd::base_strided_kernel<fixed_string_utf8_assign_kernel, 1> {
  intptr_t data_size;

  fixed_string_utf8_assign_kernel(intptr_t data_size) : data_size(data_size) {}

  void single(char *dst, char *const *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    intptr_t size = std::find(src[0], src[0] + data_size, 0) - src[0];
    *dst_obj = PyUnicode_DecodeUTF8(src[0], size, NULL);
  }
};

struct fixed_string_utf16_assign_kernel : dynd::nd::base_strided_kernel<fixed_string_utf16_assign_kernel, 1> {
  intptr_t data_size;

  fixed_string_utf16_assign_kernel(intptr_t data_size) : data_size(data_size) {}

  void single(char *dst, char *const *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    const uint16_t *char16_src = reinterpret_cast<const uint16_t *>(src[0]);
    intptr_t size = std::find(char16_src, char16_src + (data_size >> 1), 0) - char16_src;
    *dst_obj = PyUnicode_DecodeUTF16(src[0], size * 2, NULL, NULL);
  }
};

struct fixed_string_utf32_assign_kernel : dynd::nd::base_strided_kernel<fixed_string_utf32_assign_kernel, 1> {
  intptr_t data_size;

  fixed_string_utf32_assign_kernel(intptr_t data_size) : data_size(data_size) {}

  void single(char *dst, char *const *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    const uint32_t *char32_src = reinterpret_cast<const uint32_t *>(src[0]);
    intptr_t size = std::find(char32_src, char32_src + (data_size >> 2), 0) - char32_src;
    *dst_obj = PyUnicode_DecodeUTF32(src[0], size * 4, NULL, NULL);
  }
};

// TODO: Should make a more efficient strided kernel function
template <>
struct assign_to_pyobject_kernel<ndt::option_type>
    : dynd::nd::base_strided_kernel<assign_to_pyobject_kernel<ndt::option_type>, 1> {
  intptr_t m_assign_value_offset;

  ~assign_to_pyobject_kernel()
  {
    get_child()->destroy();
    get_child(m_assign_value_offset)->destroy();
  }

  void single(char *dst, char *const *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    dynd::nd::kernel_prefix *is_na = get_child();
    dynd::kernel_single_t is_na_fn = is_na->get_function<dynd::kernel_single_t>();
    dynd::nd::kernel_prefix *assign_value = get_child(m_assign_value_offset);
    dynd::kernel_single_t assign_value_fn = assign_value->get_function<dynd::kernel_single_t>();
    char value_is_na = 1;
    is_na_fn(is_na, &value_is_na, src);
    if (value_is_na == 0) {
      assign_value_fn(assign_value, dst, src);
    }
    else {
      *dst_obj = Py_None;
      Py_INCREF(*dst_obj);
    }
  }
};

template <>
struct assign_to_pyobject_kernel<ndt::type> : dynd::nd::base_strided_kernel<assign_to_pyobject_kernel<ndt::type>, 1> {
  void single(char *dst, char *const *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    dynd::ndt::type tp = *reinterpret_cast<const dynd::ndt::type *>(src[0]);
    *dst_obj = pydynd::type_from_cpp(std::move(tp));
  }
};

// TODO: Should make a more efficient strided kernel function
template <>
struct assign_to_pyobject_kernel<ndt::tuple_type>
    : dynd::nd::base_strided_kernel<assign_to_pyobject_kernel<ndt::tuple_type>, 1> {
  dynd::ndt::type src_tp;
  const char *src_arrmeta;
  std::vector<intptr_t> m_copy_el_offsets;

  assign_to_pyobject_kernel(dynd::ndt::type src_tp, const char *src_arrmeta) : src_tp(src_tp), src_arrmeta(src_arrmeta)
  {
  }

  ~assign_to_pyobject_kernel()
  {
    for (size_t i = 0; i < m_copy_el_offsets.size(); ++i) {
      get_child(m_copy_el_offsets[i])->destroy();
    }
  }

  void single(char *dst, char *const *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    intptr_t field_count = src_tp.extended<dynd::ndt::tuple_type>()->get_field_count();
    const uintptr_t *field_offsets = reinterpret_cast<const uintptr_t *>(src_arrmeta);
    pydynd::py_ref tup = pydynd::capture_if_not_null(PyTuple_New(field_count));
    for (intptr_t i = 0; i < field_count; ++i) {
      nd::kernel_prefix *copy_el = get_child(m_copy_el_offsets[i]);
      dynd::kernel_single_t copy_el_fn = copy_el->get_function<dynd::kernel_single_t>();
      char *el_src = src[0] + field_offsets[i];
      char *el_dst = reinterpret_cast<char *>(((PyTupleObject *)tup.get())->ob_item + i);
      copy_el_fn(copy_el, el_dst, &el_src);
    }
    if (PyErr_Occurred()) {
      throw std::exception();
    }
    *dst_obj = pydynd::release(std::move(tup));
  }
};

// TODO: Should make a more efficient strided kernel function
template <>
struct assign_to_pyobject_kernel<ndt::struct_type>
    : dynd::nd::base_strided_kernel<assign_to_pyobject_kernel<ndt::struct_type>, 1> {
  dynd::ndt::type m_src_tp;
  const char *m_src_arrmeta;
  std::vector<intptr_t> m_copy_el_offsets;
  pydynd::py_ref m_field_names;

  ~assign_to_pyobject_kernel()
  {
    for (size_t i = 0; i < m_copy_el_offsets.size(); ++i) {
      get_child(m_copy_el_offsets[i])->destroy();
    }
  }

  void single(char *dst, char *const *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    intptr_t field_count = m_src_tp.extended<dynd::ndt::tuple_type>()->get_field_count();
    const uintptr_t *field_offsets = reinterpret_cast<const uintptr_t *>(m_src_arrmeta);
    pydynd::py_ref dct = pydynd::capture_if_not_null(PyDict_New());
    for (intptr_t i = 0; i < field_count; ++i) {
      dynd::nd::kernel_prefix *copy_el = get_child(m_copy_el_offsets[i]);
      dynd::kernel_single_t copy_el_fn = copy_el->get_function<dynd::kernel_single_t>();
      char *el_src = src[0] + field_offsets[i];
      pydynd::py_ref el;
      copy_el_fn(copy_el, reinterpret_cast<char *>(&el), &el_src);
      PyDict_SetItem(dct.get(), PyTuple_GET_ITEM(m_field_names.get(), i), el.get());
    }
    if (PyErr_Occurred()) {
      throw std::exception();
    }
    *dst_obj = pydynd::release(std::move(dct));
  }
};

template <>
struct assign_to_pyobject_kernel<ndt::fixed_dim_type>
    : dynd::nd::base_strided_kernel<assign_to_pyobject_kernel<ndt::fixed_dim_type>, 1> {
  intptr_t dim_size, stride;

  assign_to_pyobject_kernel(intptr_t dim_size, intptr_t stride) : dim_size(dim_size), stride(stride) {}

  ~assign_to_pyobject_kernel() { get_child()->destroy(); }

  void single(char *dst, char *const *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    pydynd::py_ref lst = pydynd::capture_if_not_null(PyList_New(dim_size));
    nd::kernel_prefix *copy_el = get_child();
    dynd::kernel_strided_t copy_el_fn = copy_el->get_function<dynd::kernel_strided_t>();
    copy_el_fn(copy_el, reinterpret_cast<char *>(((PyListObject *)lst.get())->ob_item), sizeof(PyObject *), src,
               &stride, dim_size);
    if (PyErr_Occurred()) {
      throw std::exception();
    }
    *dst_obj = pydynd::release(std::move(lst));
  }
};

template <>
struct assign_to_pyobject_kernel<ndt::var_dim_type>
    : dynd::nd::base_strided_kernel<assign_to_pyobject_kernel<ndt::var_dim_type>, 1> {
  intptr_t offset, stride;

  assign_to_pyobject_kernel(intptr_t offset, intptr_t stride) : offset(offset), stride(stride) {}

  ~assign_to_pyobject_kernel() { get_child()->destroy(); }

  void single(char *dst, char *const *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    const dynd::ndt::var_dim_type::data_type *vd = reinterpret_cast<const dynd::ndt::var_dim_type::data_type *>(src[0]);
    pydynd::py_ref lst = pydynd::capture_if_not_null(PyList_New(vd->size));
    dynd::nd::kernel_prefix *copy_el = get_child();
    dynd::kernel_strided_t copy_el_fn = copy_el->get_function<dynd::kernel_strided_t>();
    char *el_src = vd->begin + offset;
    copy_el_fn(copy_el, reinterpret_cast<char *>(((PyListObject *)lst.get())->ob_item), sizeof(PyObject *), &el_src,
               &stride, vd->size);
    if (PyErr_Occurred()) {
      throw std::exception();
    }
    *dst_obj = pydynd::release(std::move(lst));
  }
};

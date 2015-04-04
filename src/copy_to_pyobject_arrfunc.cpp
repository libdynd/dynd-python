//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>
#include <datetime.h>

#include "copy_to_pyobject_arrfunc.hpp"
#include "utility_functions.hpp"
#include "type_functions.hpp"

#include <dynd/array.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/fixedbytes_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/categorical_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/time_type.hpp>
#include <dynd/types/datetime_type.hpp>
#include <dynd/types/option_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/base_tuple_type.hpp>
#include <dynd/types/base_struct_type.hpp>
#include <dynd/types/pointer_type.hpp>
#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/func/copy.hpp>
#include <dynd/kernels/chain.hpp>

using namespace std;
using namespace dynd;
using namespace pydynd;

namespace {

struct bool_ck : public kernels::unary_ck<bool_ck> {
  inline void single(char *dst, const char *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = (*src != 0) ? Py_True : Py_False;
    Py_INCREF(*dst_obj);
  }
};

PyObject *pyint_from_int(int8_t v) {
#if PY_VERSION_HEX >= 0x03000000
  return PyLong_FromLong(v);
#else
  return PyInt_FromLong(v);
#endif
}

PyObject *pyint_from_int(uint8_t v) {
#if PY_VERSION_HEX >= 0x03000000
  return PyLong_FromLong(v);
#else
  return PyInt_FromLong(v);
#endif
}

PyObject *pyint_from_int(int16_t v) {
#if PY_VERSION_HEX >= 0x03000000
  return PyLong_FromLong(v);
#else
  return PyInt_FromLong(v);
#endif
}

PyObject *pyint_from_int(uint16_t v) {
#if PY_VERSION_HEX >= 0x03000000
  return PyLong_FromLong(v);
#else
  return PyInt_FromLong(v);
#endif
}

PyObject *pyint_from_int(int32_t v) {
#if PY_VERSION_HEX >= 0x03000000
  return PyLong_FromLong(v);
#else
  return PyInt_FromLong(v);
#endif
}

PyObject *pyint_from_int(uint32_t v) {
  return PyLong_FromUnsignedLong(v);
}

#if SIZEOF_LONG == 8
PyObject *pyint_from_int(int64_t v) {
#if PY_VERSION_HEX >= 0x03000000
  return PyLong_FromLong(v);
#else
  return PyInt_FromLong(v);
#endif
}

PyObject *pyint_from_int(uint64_t v) {
  return PyLong_FromUnsignedLong(v);
}
#else
PyObject *pyint_from_int(int64_t v) {
  return PyLong_FromLongLong(v);
}

PyObject *pyint_from_int(uint64_t v) {
  return PyLong_FromUnsignedLongLong(v);
}
#endif

PyObject *pyint_from_int(const dynd_uint128& val)
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

PyObject *pyint_from_int(const dynd_int128& val)
{
  if (val.is_negative()) {
    if (val.m_hi == 0xffffffffffffffffULL &&
        (val.m_hi & 0x8000000000000000ULL) != 0) {
      return PyLong_FromLongLong(static_cast<int64_t>(val.m_lo));
    }
    pyobject_ownref absval(pyint_from_int(static_cast<dynd_uint128>(-val)));
    return PyNumber_Negative(absval.get());
  } else {
    return pyint_from_int(static_cast<dynd_uint128>(val));
  }
}

template <class T>
struct int_ck : public kernels::unary_ck<int_ck<T> > {
  inline void single(char *dst, const char *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    *dst_obj = pyint_from_int(*reinterpret_cast<const T *>(src));
  }
};

template<class T>
struct float_ck : public kernels::unary_ck<float_ck<T> > {
  inline void single(char *dst, const char *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    *dst_obj = PyFloat_FromDouble(*reinterpret_cast<const T *>(src));
  }
};

template<class T>
struct complex_float_ck : public kernels::unary_ck<complex_float_ck<T> > {
  inline void single(char *dst, const char *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    const dynd::complex<T> *val = reinterpret_cast<const dynd::complex<T> *>(src);
    *dst_obj = PyComplex_FromDoubles(val->real(), val->imag());
  }
};

struct bytes_ck : public kernels::unary_ck<bytes_ck> {
  inline void single(char *dst, const char *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    const bytes_type_data *bd = reinterpret_cast<const bytes_type_data *>(src);
    *dst_obj = PyBytes_FromStringAndSize(bd->begin, bd->end - bd->begin);
  }
};

struct fixedbytes_ck : public kernels::unary_ck<fixedbytes_ck> {
  intptr_t m_data_size;

  inline void single(char *dst, const char *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    *dst_obj = PyBytes_FromStringAndSize(src, m_data_size);
  }
};

struct char_ck : public kernels::unary_ck<char_ck> {
  inline void single(char *dst, const char *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    *dst_obj = PyUnicode_DecodeUTF32(src, 4, NULL, NULL);
  }
};

struct string_ascii_ck : public kernels::unary_ck<string_ascii_ck> {
  inline void single(char *dst, const char *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    const string_type_data *sd =
        reinterpret_cast<const string_type_data *>(src);
    *dst_obj = PyUnicode_DecodeASCII(sd->begin, sd->end - sd->begin, NULL);
  }
};

struct string_utf8_ck : public kernels::unary_ck<string_utf8_ck> {
  inline void single(char *dst, const char *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    const string_type_data *sd =
        reinterpret_cast<const string_type_data *>(src);
    *dst_obj = PyUnicode_DecodeUTF8(sd->begin, sd->end - sd->begin, NULL);
  }
};

struct string_utf16_ck : public kernels::unary_ck<string_utf16_ck> {
  inline void single(char *dst, const char *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    const string_type_data *sd =
        reinterpret_cast<const string_type_data *>(src);
    *dst_obj =
        PyUnicode_DecodeUTF16(sd->begin, sd->end - sd->begin, NULL, NULL);
  }
};

struct string_utf32_ck : public kernels::unary_ck<string_utf32_ck> {
  inline void single(char *dst, const char *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    const string_type_data *sd =
        reinterpret_cast<const string_type_data *>(src);
    *dst_obj =
        PyUnicode_DecodeUTF32(sd->begin, sd->end - sd->begin, NULL, NULL);
  }
};

struct fixedstring_ascii_ck : public kernels::unary_ck<fixedstring_ascii_ck> {
  intptr_t m_data_size;

  inline void single(char *dst, const char *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    intptr_t size = find(src, src + m_data_size, 0) - src;
    *dst_obj = PyUnicode_DecodeASCII(src, size, NULL);
  }
};

struct fixedstring_utf8_ck : public kernels::unary_ck<fixedstring_utf8_ck> {
  intptr_t m_data_size;

  inline void single(char *dst, const char *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    intptr_t size = find(src, src + m_data_size, 0) - src;
    *dst_obj = PyUnicode_DecodeUTF8(src, size, NULL);
  }
};

struct fixedstring_utf16_ck : public kernels::unary_ck<fixedstring_utf16_ck> {
  intptr_t m_data_size;

  inline void single(char *dst, const char *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    const uint16_t *char16_src = reinterpret_cast<const uint16_t *>(src);
    intptr_t size =
        find(char16_src, char16_src + (m_data_size >> 1), 0) - char16_src;
    *dst_obj = PyUnicode_DecodeUTF16(src, size * 2, NULL, NULL);
  }
};

struct fixedstring_utf32_ck : public kernels::unary_ck<fixedstring_utf32_ck> {
  intptr_t m_data_size;

  inline void single(char *dst, const char *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    const uint32_t *char32_src = reinterpret_cast<const uint32_t *>(src);
    intptr_t size =
        find(char32_src, char32_src + (m_data_size >> 2), 0) - char32_src;
    *dst_obj = PyUnicode_DecodeUTF32(src, size * 4, NULL, NULL);
  }
};

struct date_ck : public kernels::unary_ck<date_ck> {
  ndt::type m_src_tp;
  const char *m_src_arrmeta;

  inline void single(char *dst, const char *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    const date_type *dd = m_src_tp.extended<date_type>();
    date_ymd ymd = dd->get_ymd(m_src_arrmeta, src);
    *dst_obj = PyDate_FromDate(ymd.year, ymd.month, ymd.day);
  }
};

struct time_ck : public kernels::unary_ck<time_ck> {
  ndt::type m_src_tp;
  const char *m_src_arrmeta;

  inline void single(char *dst, const char *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    const time_type *tt = m_src_tp.extended<time_type>();
    time_hmst hmst = tt->get_time(m_src_arrmeta, src);
    *dst_obj = PyTime_FromTime(hmst.hour, hmst.minute, hmst.second,
                               hmst.tick / DYND_TICKS_PER_MICROSECOND);
  }
};

struct datetime_ck : public kernels::unary_ck<datetime_ck> {
  ndt::type m_src_tp;
  const char *m_src_arrmeta;

  inline void single(char *dst, const char *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    const datetime_type *dd = m_src_tp.extended<datetime_type>();
    int32_t year, month, day, hour, minute, second, tick;
    dd->get_cal(m_src_arrmeta, src, year, month, day, hour, minute, second,
                tick);
    int32_t usecond = tick / 10;
    *dst_obj = PyDateTime_FromDateAndTime(year, month, day, hour, minute,
                                          second, usecond);
  }
};

struct type_ck : public kernels::unary_ck<type_ck> {
  inline void single(char *dst, const char *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    ndt::type tp(reinterpret_cast<const type_type_data *>(src)->tp, true);
    *dst_obj = wrap_ndt_type(std::move(tp));
  }
};

// TODO: Should make a more efficient strided kernel function
struct option_ck : public kernels::unary_ck<option_ck> {
  intptr_t m_copy_value_offset;

  inline void single(char *dst, char *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    ckernel_prefix *is_avail = get_child_ckernel();
    expr_single_t is_avail_fn = is_avail->get_function<expr_single_t>();
    ckernel_prefix *copy_value = get_child_ckernel(m_copy_value_offset);
    expr_single_t copy_value_fn = copy_value->get_function<expr_single_t>();
    char value_is_avail = 0;
    is_avail_fn(&value_is_avail, &src, is_avail);
    if (value_is_avail != 0) {
      copy_value_fn(dst, &src, copy_value);
    } else {
      *dst_obj = Py_None;
      Py_INCREF(*dst_obj);
    }
  }

  inline void destruct_children()
  {
    get_child_ckernel()->destroy();
    base.destroy_child_ckernel(m_copy_value_offset);
  }
};

struct strided_ck : public kernels::unary_ck<strided_ck> {
  intptr_t m_dim_size, m_stride;
  inline void single(char *dst, char *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    pyobject_ownref lst(PyList_New(m_dim_size));
    ckernel_prefix *copy_el = get_child_ckernel();
    expr_strided_t copy_el_fn = copy_el->get_function<expr_strided_t>();
    copy_el_fn(reinterpret_cast<char *>(((PyListObject *)lst.get())->ob_item),
      sizeof(PyObject *), &src, &m_stride, m_dim_size, copy_el);
    if (PyErr_Occurred()) {
      throw std::exception();
    }
    *dst_obj = lst.release();
  }

  inline void destruct_children()
  {
    get_child_ckernel()->destroy();
  }
};

struct var_dim_ck : public kernels::unary_ck<var_dim_ck> {
  intptr_t m_offset, m_stride;

  inline void single(char *dst, char *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    const var_dim_type_data *vd =
        reinterpret_cast<const var_dim_type_data *>(src);
    pyobject_ownref lst(PyList_New(vd->size));
    ckernel_prefix *copy_el = get_child_ckernel();
    expr_strided_t copy_el_fn = copy_el->get_function<expr_strided_t>();
    char *el_src = vd->begin + m_offset;
    copy_el_fn(reinterpret_cast<char *>(((PyListObject *)lst.get())->ob_item),
               sizeof(PyObject *), &el_src, &m_stride, vd->size, copy_el);
    if (PyErr_Occurred()) {
      throw std::exception();
    }
    *dst_obj = lst.release();
  }

  inline void destruct_children()
  {
    get_child_ckernel()->destroy();
  }
};

// TODO: Should make a more efficient strided kernel function
struct struct_ck : public kernels::unary_ck<struct_ck> {
  ndt::type m_src_tp;
  const char *m_src_arrmeta;
  vector<intptr_t> m_copy_el_offsets;
  pyobject_ownref m_field_names;

  inline void single(char *dst, char *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    intptr_t field_count = m_src_tp.extended<base_tuple_type>()->get_field_count();
    const uintptr_t *field_offsets =
        m_src_tp.extended<base_tuple_type>()->get_data_offsets(m_src_arrmeta);
    pyobject_ownref dct(PyDict_New());
    for (intptr_t i = 0; i < field_count; ++i) {
      ckernel_prefix *copy_el = get_child_ckernel(m_copy_el_offsets[i]);
      expr_single_t copy_el_fn = copy_el->get_function<expr_single_t>();
      char *el_src = src + field_offsets[i];
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

  inline void destruct_children()
  {
    for (size_t i = 0; i < m_copy_el_offsets.size(); ++i) {
      base.destroy_child_ckernel(m_copy_el_offsets[i]);
    }
  }
};

// TODO: Should make a more efficient strided kernel function
struct tuple_ck : public kernels::unary_ck<tuple_ck> {
  ndt::type m_src_tp;
  const char *m_src_arrmeta;
  vector<intptr_t> m_copy_el_offsets;

  inline void single(char *dst, char *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    intptr_t field_count = m_src_tp.extended<base_tuple_type>()->get_field_count();
    const uintptr_t *field_offsets =
        m_src_tp.extended<base_tuple_type>()->get_data_offsets(m_src_arrmeta);
    pyobject_ownref tup(PyTuple_New(field_count));
    for (intptr_t i = 0; i < field_count; ++i) {
      ckernel_prefix *copy_el = get_child_ckernel(m_copy_el_offsets[i]);
      expr_single_t copy_el_fn = copy_el->get_function<expr_single_t>();
      char *el_src = src + field_offsets[i];
      char *el_dst =
          reinterpret_cast<char *>(((PyTupleObject *)tup.get())->ob_item + i);
      copy_el_fn(el_dst, &el_src, copy_el);
    }
    if (PyErr_Occurred()) {
      throw std::exception();
    }
    *dst_obj = tup.release();
  }

  inline void destruct_children()
  {
    for (size_t i = 0; i < m_copy_el_offsets.size(); ++i) {
      base.destroy_child_ckernel(m_copy_el_offsets[i]);
    }
  }
};

struct pointer_ck : public kernels::unary_ck<pointer_ck> {
  inline void single(char *dst, char *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    ckernel_prefix *copy_value = get_child_ckernel();
    expr_single_t copy_value_fn = copy_value->get_function<expr_single_t>();
    // The src value is a pointer, and copy_value_fn expects a pointer
    // to that pointer
    char **src_ptr = reinterpret_cast<char **>(src);
    copy_value_fn(dst, src_ptr, copy_value);
  }

  inline void destruct_children()
  {
    get_child_ckernel()->destroy();
  }
};

} // anonymous namespace

static intptr_t instantiate_copy_to_pyobject(
    const arrfunc_type_data *self_af, const arrfunc_type *af_tp,
    char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *dst_arrmeta, intptr_t nsrc, const ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx, const nd::array &kwds, const std::map<nd::string, ndt::type> &tp_vars)
{
  if (dst_tp.get_type_id() != void_type_id) {
    stringstream ss;
    ss << "Cannot instantiate arrfunc with signature ";
    ss << af_tp << " with types (";
    ss << src_tp[0] << ") -> " << dst_tp;
    throw type_error(ss.str());
  }

  if (!kwds.is_null()) {
    throw invalid_argument("unexpected non-NULL kwds value to "
                           "copy_to_pyobject instantiation");
  }

  bool struct_as_pytuple = *self_af->get_data_as<bool>();

  switch (src_tp[0].get_type_id()) {
  case bool_type_id:
    bool_ck::create_leaf(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case int8_type_id:
    int_ck<int8_t>::create_leaf(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case int16_type_id:
    int_ck<int16_t>::create_leaf(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case int32_type_id:
    int_ck<int32_t>::create_leaf(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case int64_type_id:
    int_ck<int64_t>::create_leaf(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case int128_type_id:
    int_ck<dynd_int128>::create_leaf(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case uint8_type_id:
    int_ck<uint8_t>::create_leaf(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case uint16_type_id:
    int_ck<uint16_t>::create_leaf(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case uint32_type_id:
    int_ck<uint32_t>::create_leaf(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case uint64_type_id:
    int_ck<uint64_t>::create_leaf(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case uint128_type_id:
    int_ck<dynd_uint128>::create_leaf(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case float16_type_id:
    float_ck<dynd_float16>::create_leaf(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case float32_type_id:
    float_ck<float>::create_leaf(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case float64_type_id:
    float_ck<double>::create_leaf(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case complex_float32_type_id:
    complex_float_ck<float>::create_leaf(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case complex_float64_type_id:
    complex_float_ck<double>::create_leaf(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case bytes_type_id:
    bytes_ck::create_leaf(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case fixedbytes_type_id: {
    fixedbytes_ck *self = fixedbytes_ck::create_leaf(ckb, kernreq, ckb_offset);
    self->m_data_size = src_tp[0].extended<fixedbytes_type>()->get_data_size();
    return ckb_offset;
  }
  case char_type_id:
    char_ck::create_leaf(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case string_type_id:
    switch (src_tp[0].extended<base_string_type>()->get_encoding()) {
    case string_encoding_ascii:
      string_ascii_ck::create_leaf(ckb, kernreq, ckb_offset);
      return ckb_offset;
    case string_encoding_utf_8:
      string_utf8_ck::create_leaf(ckb, kernreq, ckb_offset);
      return ckb_offset;
    case string_encoding_ucs_2:
    case string_encoding_utf_16:
      string_utf16_ck::create_leaf(ckb, kernreq, ckb_offset);
      return ckb_offset;
    case string_encoding_utf_32:
      string_utf32_ck::create_leaf(ckb, kernreq, ckb_offset);
      return ckb_offset;
    default:
      break;
    }
    break;
  case fixedstring_type_id:
    switch (src_tp[0].extended<base_string_type>()->get_encoding()) {
    case string_encoding_ascii: {
      fixedstring_ascii_ck *self =
          fixedstring_ascii_ck::create_leaf(ckb, kernreq, ckb_offset);
      self->m_data_size = src_tp[0].get_data_size();
      return ckb_offset;
    }
    case string_encoding_utf_8: {
      fixedstring_utf8_ck *self =
          fixedstring_utf8_ck::create_leaf(ckb, kernreq, ckb_offset);
      self->m_data_size = src_tp[0].get_data_size();
      return ckb_offset;
    }
    case string_encoding_ucs_2:
    case string_encoding_utf_16: {
      fixedstring_utf16_ck *self =
          fixedstring_utf16_ck::create_leaf(ckb, kernreq, ckb_offset);
      self->m_data_size = src_tp[0].get_data_size();
      return ckb_offset;
    }
    case string_encoding_utf_32: {
      fixedstring_utf32_ck *self =
          fixedstring_utf32_ck::create_leaf(ckb, kernreq, ckb_offset);
      self->m_data_size = src_tp[0].get_data_size();
      return ckb_offset;
    }
    default:
      break;
    }
    break;
  case categorical_type_id: {
    // Assign via an intermediate category_type buffer
    const ndt::type &buf_tp =
        src_tp[0].extended<categorical_type>()->get_category_type();
    nd::arrfunc copy_af =
        make_arrfunc_from_assignment(buf_tp, src_tp[0], assign_error_default);
    return nd::functional::make_chain_buf_tp_ckernel(
        copy_af.get(), copy_af.get_type(), self_af, af_tp, buf_tp, ckb, ckb_offset,
        dst_tp, dst_arrmeta, src_tp, src_arrmeta, kernreq, ectx);
  }
  case date_type_id: {
    date_ck *self = date_ck::create_leaf(ckb, kernreq, ckb_offset);
    self->m_src_tp = src_tp[0];
    self->m_src_arrmeta = src_arrmeta[0];
    return ckb_offset;
  }
  case time_type_id: {
    time_ck *self = time_ck::create_leaf(ckb, kernreq, ckb_offset);
    self->m_src_tp = src_tp[0];
    self->m_src_arrmeta = src_arrmeta[0];
    return ckb_offset;
  }
  case datetime_type_id: {
    datetime_ck *self = datetime_ck::create_leaf(ckb, kernreq, ckb_offset);
    self->m_src_tp = src_tp[0];
    self->m_src_arrmeta = src_arrmeta[0];
    return ckb_offset;
  }
  case type_type_id:
    type_ck::create_leaf(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case option_type_id: {
    intptr_t root_ckb_offset = ckb_offset;
    option_ck *self = option_ck::create(ckb, kernreq, ckb_offset);
    const arrfunc_type_data *is_avail_af =
        src_tp[0].extended<option_type>()->get_is_avail_arrfunc();
    const arrfunc_type *is_avail_af_tp =
        src_tp[0].extended<option_type>()->get_is_avail_arrfunc_type();
    ckb_offset = is_avail_af->instantiate(
        is_avail_af, is_avail_af_tp, NULL, ckb, ckb_offset,
        ndt::make_type<dynd_bool>(), NULL, nsrc, src_tp, src_arrmeta,
        kernel_request_single, ectx, nd::array(), tp_vars);
    reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->ensure_capacity(ckb_offset);
    self = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->get_at<option_ck>(root_ckb_offset);
    self->m_copy_value_offset = ckb_offset - root_ckb_offset;
    ndt::type src_value_tp = src_tp[0].extended<option_type>()->get_value_type();
    ckb_offset = self_af->instantiate(
        self_af, af_tp, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, &src_value_tp,
        src_arrmeta, kernel_request_single, ectx, nd::array(), tp_vars);
    return ckb_offset;
  }
  case fixed_dim_type_id:
  case cfixed_dim_type_id: {
    intptr_t dim_size, stride;
    ndt::type el_tp;
    const char *el_arrmeta;
    if (src_tp[0].get_as_strided(src_arrmeta[0], &dim_size, &stride, &el_tp,
                                 &el_arrmeta)) {
      strided_ck *self = strided_ck::create(ckb, kernreq, ckb_offset);
      self->m_dim_size = dim_size;
      self->m_stride = stride;
      return self_af->instantiate(
          self_af, af_tp, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, &el_tp,
          &el_arrmeta, kernel_request_strided, ectx, nd::array(), tp_vars);
    }
    break;
  }
  case var_dim_type_id: {
    var_dim_ck *self = var_dim_ck::create(ckb, kernreq, ckb_offset);
    self->m_offset =
        reinterpret_cast<const var_dim_type_arrmeta *>(src_arrmeta[0])->offset;
    self->m_stride =
        reinterpret_cast<const var_dim_type_arrmeta *>(src_arrmeta[0])->stride;
    ndt::type el_tp = src_tp[0].extended<var_dim_type>()->get_element_type();
    const char *el_arrmeta = src_arrmeta[0] + sizeof(var_dim_type_arrmeta);
    return self_af->instantiate(
        self_af, af_tp, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, &el_tp,
        &el_arrmeta, kernel_request_strided, ectx, nd::array(), tp_vars);
  }
  case cstruct_type_id:
  case struct_type_id:
    if (!struct_as_pytuple) {
      intptr_t root_ckb_offset = ckb_offset;
      struct_ck *self = struct_ck::create(ckb, kernreq, ckb_offset);
      self->m_src_tp = src_tp[0];
      self->m_src_arrmeta = src_arrmeta[0];
      intptr_t field_count = src_tp[0].extended<base_struct_type>()->get_field_count();
      const ndt::type *field_types =
          src_tp[0].extended<base_struct_type>()->get_field_types_raw();
      const uintptr_t *arrmeta_offsets =
          src_tp[0].extended<base_struct_type>()->get_arrmeta_offsets_raw();
      self->m_field_names.reset(PyTuple_New(field_count));
      for (intptr_t i = 0; i < field_count; ++i) {
        const string_type_data &rawname =
            src_tp[0].extended<base_struct_type>()->get_field_name_raw(i);
        pyobject_ownref name(PyUnicode_DecodeUTF8(
            rawname.begin, rawname.end - rawname.begin, NULL));
        PyTuple_SET_ITEM(self->m_field_names.get(), i, name.release());
      }
      self->m_copy_el_offsets.resize(field_count);
      for (intptr_t i = 0; i < field_count; ++i) {
        reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->ensure_capacity(ckb_offset);
        self = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->get_at<struct_ck>(root_ckb_offset);
        self->m_copy_el_offsets[i] = ckb_offset - root_ckb_offset;
        const char *field_arrmeta = src_arrmeta[0] + arrmeta_offsets[i];
        ckb_offset = self_af->instantiate(self_af, af_tp, NULL, ckb, ckb_offset,
                                          dst_tp, dst_arrmeta, nsrc, &field_types[i],
                                          &field_arrmeta, kernel_request_single,
                                          ectx, nd::array(), tp_vars);
      }
      return ckb_offset;
    }
    // Otherwise fall through to the tuple case
  case ctuple_type_id:
  case tuple_type_id: {
    intptr_t root_ckb_offset = ckb_offset;
    tuple_ck *self = tuple_ck::create(ckb, kernreq, ckb_offset);
    self->m_src_tp = src_tp[0];
    self->m_src_arrmeta = src_arrmeta[0];
    intptr_t field_count = src_tp[0].extended<base_tuple_type>()->get_field_count();
    const ndt::type *field_types =
        src_tp[0].extended<base_tuple_type>()->get_field_types_raw();
    const uintptr_t *arrmeta_offsets =
        src_tp[0].extended<base_tuple_type>()->get_arrmeta_offsets_raw();
    self->m_copy_el_offsets.resize(field_count);
    for (intptr_t i = 0; i < field_count; ++i) {
      reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->ensure_capacity(ckb_offset);
      self = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->get_at<tuple_ck>(root_ckb_offset);
      self->m_copy_el_offsets[i] = ckb_offset - root_ckb_offset;
      const char *field_arrmeta = src_arrmeta[0] + arrmeta_offsets[i];
      ckb_offset = self_af->instantiate(self_af, af_tp, NULL, ckb, ckb_offset, dst_tp,
                                        dst_arrmeta, nsrc, &field_types[i],
                                        &field_arrmeta, kernel_request_single,
                                        ectx, nd::array(), tp_vars);
    }
    return ckb_offset;
  }
  case pointer_type_id: {
    pointer_ck *self = pointer_ck::create(ckb, kernreq, ckb_offset);
    ndt::type src_value_tp = src_tp[0].extended<pointer_type>()->get_target_type();
    return self_af->instantiate(
        self_af, af_tp, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc, &src_value_tp,
        src_arrmeta, kernel_request_single, ectx, nd::array(), tp_vars);
  }
  default:
    break;
  }

  if (src_tp[0].get_kind() == expr_kind) {
    return nd::functional::make_chain_buf_tp_ckernel(
        nd::copy.get(), nd::copy.get_type(), self_af,
        af_tp, src_tp[0].value_type(), ckb, ckb_offset, dst_tp, dst_arrmeta,
        src_tp, src_arrmeta, kernreq, ectx);
  }

  stringstream ss;
  ss << "Unable to copy dynd value with type " << src_tp[0]
     << " to a Python object";
  throw invalid_argument(ss.str());
}

static nd::arrfunc make_copy_to_pyobject_arrfunc(bool struct_as_pytuple)
{
  nd::array out_af = nd::empty("(Any) -> void");
  arrfunc_type_data *af =
      reinterpret_cast<arrfunc_type_data *>(out_af.get_readwrite_originptr());
  af->instantiate = &instantiate_copy_to_pyobject;
  *af->get_data_as<bool>() = struct_as_pytuple;
  out_af.flag_as_immutable();
  return out_af;
}

dynd::nd::arrfunc pydynd::copy_to_pyobject_dict::make() {
  PyDateTime_IMPORT;
  return make_copy_to_pyobject_arrfunc(false);
}

dynd::nd::arrfunc pydynd::copy_to_pyobject_tuple::make() {
  PyDateTime_IMPORT;
  return make_copy_to_pyobject_arrfunc(true);
}

struct pydynd::copy_to_pyobject_dict pydynd::copy_to_pyobject_dict;
struct pydynd::copy_to_pyobject_tuple pydynd::copy_to_pyobject_tuple;
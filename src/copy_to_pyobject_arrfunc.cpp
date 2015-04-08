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
#include <dynd/func/chain.hpp>

#include "kernels/copy_kernel.hpp"

using namespace std;
using namespace pydynd;

static dynd::nd::arrfunc make_copy_to_pyobject_arrfunc(bool struct_as_pytuple);

namespace {

// TODO: Should make a more efficient strided kernel function
struct option_ck : public dynd::kernels::unary_ck<option_ck> {
  intptr_t m_copy_value_offset;

  inline void single(char *dst, char *src)
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

struct strided_ck : public dynd::kernels::unary_ck<strided_ck> {
  intptr_t m_dim_size, m_stride;
  inline void single(char *dst, char *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    pyobject_ownref lst(PyList_New(m_dim_size));
    dynd::ckernel_prefix *copy_el = get_child_ckernel();
    expr_strided_t copy_el_fn = copy_el->get_function<expr_strided_t>();
    copy_el_fn(reinterpret_cast<char *>(((PyListObject *)lst.get())->ob_item),
               sizeof(PyObject *), &src, &m_stride, m_dim_size, copy_el);
    if (PyErr_Occurred()) {
      throw std::exception();
    }
    *dst_obj = lst.release();
  }

  inline void destruct_children() { get_child_ckernel()->destroy(); }
};

struct var_dim_ck : public dynd::kernels::unary_ck<var_dim_ck> {
  intptr_t m_offset, m_stride;

  inline void single(char *dst, char *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    const var_dim_type_data *vd =
        reinterpret_cast<const var_dim_type_data *>(src);
    pyobject_ownref lst(PyList_New(vd->size));
    dynd::ckernel_prefix *copy_el = get_child_ckernel();
    expr_strided_t copy_el_fn = copy_el->get_function<expr_strided_t>();
    char *el_src = vd->begin + m_offset;
    copy_el_fn(reinterpret_cast<char *>(((PyListObject *)lst.get())->ob_item),
               sizeof(PyObject *), &el_src, &m_stride, vd->size, copy_el);
    if (PyErr_Occurred()) {
      throw std::exception();
    }
    *dst_obj = lst.release();
  }

  inline void destruct_children() { get_child_ckernel()->destroy(); }
};

// TODO: Should make a more efficient strided kernel function
struct struct_ck : public dynd::kernels::unary_ck<struct_ck> {
  dynd::ndt::type m_src_tp;
  const char *m_src_arrmeta;
  vector<intptr_t> m_copy_el_offsets;
  pyobject_ownref m_field_names;

  inline void single(char *dst, char *src)
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
struct tuple_ck : public dynd::kernels::unary_ck<tuple_ck> {
  dynd::ndt::type m_src_tp;
  const char *m_src_arrmeta;
  vector<intptr_t> m_copy_el_offsets;

  inline void single(char *dst, char *src)
  {
    PyObject **dst_obj = reinterpret_cast<PyObject **>(dst);
    Py_XDECREF(*dst_obj);
    *dst_obj = NULL;
    intptr_t field_count =
        m_src_tp.extended<base_tuple_type>()->get_field_count();
    const uintptr_t *field_offsets =
        m_src_tp.extended<base_tuple_type>()->get_data_offsets(m_src_arrmeta);
    pyobject_ownref tup(PyTuple_New(field_count));
    for (intptr_t i = 0; i < field_count; ++i) {
      dynd::ckernel_prefix *copy_el = get_child_ckernel(m_copy_el_offsets[i]);
      dynd::expr_single_t copy_el_fn =
          copy_el->get_function<dynd::expr_single_t>();
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

} // anonymous namespace

static intptr_t instantiate_copy_to_pyobject(
    const arrfunc_type_data *self_af, const arrfunc_type *af_tp, char *data,
    void *ckb, intptr_t ckb_offset, const dynd::ndt::type &dst_tp,
    const char *dst_arrmeta, intptr_t nsrc, const dynd::ndt::type *src_tp,
    const char *const *src_arrmeta, kernel_request_t kernreq,
    const eval::eval_context *ectx, const dynd::nd::array &kwds,
    const std::map<dynd::nd::string, dynd::ndt::type> &tp_vars)
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
    return pydynd::nd::copy_kernel<bool_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case int8_type_id:
    return pydynd::nd::copy_kernel<int8_type_id>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case int16_type_id:
    pydynd::nd::copy_kernel<int16_type_id>::create(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case int32_type_id:
    pydynd::nd::copy_kernel<int32_type_id>::create(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case int64_type_id:
    pydynd::nd::copy_kernel<int64_type_id>::create(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case int128_type_id:
    pydynd::nd::copy_int_kernel<dynd::dynd_int128>::create(ckb, kernreq,
                                                           ckb_offset);
    return ckb_offset;
  case uint8_type_id:
    pydynd::nd::copy_int_kernel<uint8_t>::create(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case uint16_type_id:
    pydynd::nd::copy_int_kernel<uint16_t>::create(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case uint32_type_id:
    pydynd::nd::copy_int_kernel<uint32_t>::create(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case uint64_type_id:
    pydynd::nd::copy_int_kernel<uint64_t>::create(ckb, kernreq, ckb_offset);
    return ckb_offset;
  case uint128_type_id:
    return pydynd::nd::copy_int_kernel<dynd::dynd_uint128>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case float16_type_id:
    return pydynd::nd::float_copy_kernel<dynd::dynd_float16>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case float32_type_id:
    return pydynd::nd::float_copy_kernel<float>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case float64_type_id:
    return pydynd::nd::float_copy_kernel<double>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case complex_float32_type_id:
    return pydynd::nd::complex_float_copy_kernel<float>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case complex_float64_type_id:
    return pydynd::nd::complex_float_copy_kernel<double>::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case bytes_type_id:
    return pydynd::nd::bytes_copy_kernel::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case fixedbytes_type_id:
    return pydynd::nd::fixed_bytes_copy_kernel::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case char_type_id:
    return pydynd::nd::char_copy_kernel::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case string_type_id:
    return pydynd::nd::string_copy_kernel::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case fixedstring_type_id:
    return pydynd::nd::fixed_string_copy_kernel::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case categorical_type_id: {
    // Assign via an intermediate category_type buffer
    const dynd::ndt::type &buf_tp =
        src_tp[0].extended<categorical_type>()->get_category_type();
    dynd::nd::arrfunc copy_af =
        make_arrfunc_from_assignment(buf_tp, src_tp[0], assign_error_default);
    dynd::nd::arrfunc af = dynd::nd::functional::chain(
        copy_af, make_copy_to_pyobject_arrfunc(struct_as_pytuple), buf_tp);
    return af.get()->instantiate(af.get(), af.get_type(), NULL, ckb, ckb_offset,
                                 dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta,
                                 kernreq, ectx, kwds, tp_vars);
  }
  case date_type_id:
    return pydynd::nd::date_copy_kernel::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case time_type_id:
    return pydynd::nd::time_copy_kernel::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case datetime_type_id:
    return pydynd::nd::datetime_copy_kernel::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case type_type_id:
    return pydynd::nd::type_copy_kernel::instantiate(
        self_af, af_tp, data, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
        src_tp, src_arrmeta, kernreq, ectx, kwds, tp_vars);
  case option_type_id: {
    intptr_t root_ckb_offset = ckb_offset;
    option_ck *self = option_ck::create(ckb, kernreq, ckb_offset);
    const arrfunc_type_data *is_avail_af =
        src_tp[0].extended<option_type>()->get_is_avail_arrfunc();
    const arrfunc_type *is_avail_af_tp =
        src_tp[0].extended<option_type>()->get_is_avail_arrfunc_type();
    ckb_offset = is_avail_af->instantiate(
        is_avail_af, is_avail_af_tp, NULL, ckb, ckb_offset,
        dynd::ndt::make_type<dynd_bool>(), NULL, nsrc, src_tp, src_arrmeta,
        kernel_request_single, ectx, dynd::nd::array(), tp_vars);
    reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
        ->ensure_capacity(ckb_offset);
    self = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
               ->get_at<option_ck>(root_ckb_offset);
    self->m_copy_value_offset = ckb_offset - root_ckb_offset;
    dynd::ndt::type src_value_tp =
        src_tp[0].extended<option_type>()->get_value_type();
    ckb_offset = self_af->instantiate(self_af, af_tp, NULL, ckb, ckb_offset,
                                      dst_tp, dst_arrmeta, nsrc, &src_value_tp,
                                      src_arrmeta, kernel_request_single, ectx,
                                      dynd::nd::array(), tp_vars);
    return ckb_offset;
  }
  case fixed_dim_type_id:
  case cfixed_dim_type_id: {
    intptr_t dim_size, stride;
    dynd::ndt::type el_tp;
    const char *el_arrmeta;
    if (src_tp[0].get_as_strided(src_arrmeta[0], &dim_size, &stride, &el_tp,
                                 &el_arrmeta)) {
      strided_ck *self = strided_ck::create(ckb, kernreq, ckb_offset);
      self->m_dim_size = dim_size;
      self->m_stride = stride;
      return self_af->instantiate(self_af, af_tp, NULL, ckb, ckb_offset, dst_tp,
                                  dst_arrmeta, nsrc, &el_tp, &el_arrmeta,
                                  kernel_request_strided, ectx,
                                  dynd::nd::array(), tp_vars);
    }
    break;
  }
  case var_dim_type_id: {
    var_dim_ck *self = var_dim_ck::create(ckb, kernreq, ckb_offset);
    self->m_offset =
        reinterpret_cast<const var_dim_type_arrmeta *>(src_arrmeta[0])->offset;
    self->m_stride =
        reinterpret_cast<const var_dim_type_arrmeta *>(src_arrmeta[0])->stride;
    dynd::ndt::type el_tp =
        src_tp[0].extended<var_dim_type>()->get_element_type();
    const char *el_arrmeta = src_arrmeta[0] + sizeof(var_dim_type_arrmeta);
    return self_af->instantiate(self_af, af_tp, NULL, ckb, ckb_offset, dst_tp,
                                dst_arrmeta, nsrc, &el_tp, &el_arrmeta,
                                kernel_request_strided, ectx, dynd::nd::array(),
                                tp_vars);
  }
  case cstruct_type_id:
  case struct_type_id:
    if (!struct_as_pytuple) {
      intptr_t root_ckb_offset = ckb_offset;
      struct_ck *self = struct_ck::create(ckb, kernreq, ckb_offset);
      self->m_src_tp = src_tp[0];
      self->m_src_arrmeta = src_arrmeta[0];
      intptr_t field_count =
          src_tp[0].extended<base_struct_type>()->get_field_count();
      const dynd::ndt::type *field_types =
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
        reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
            ->ensure_capacity(ckb_offset);
        self = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
                   ->get_at<struct_ck>(root_ckb_offset);
        self->m_copy_el_offsets[i] = ckb_offset - root_ckb_offset;
        const char *field_arrmeta = src_arrmeta[0] + arrmeta_offsets[i];
        ckb_offset = self_af->instantiate(
            self_af, af_tp, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
            &field_types[i], &field_arrmeta, kernel_request_single, ectx,
            dynd::nd::array(), tp_vars);
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
    intptr_t field_count =
        src_tp[0].extended<base_tuple_type>()->get_field_count();
    const dynd::ndt::type *field_types =
        src_tp[0].extended<base_tuple_type>()->get_field_types_raw();
    const uintptr_t *arrmeta_offsets =
        src_tp[0].extended<base_tuple_type>()->get_arrmeta_offsets_raw();
    self->m_copy_el_offsets.resize(field_count);
    for (intptr_t i = 0; i < field_count; ++i) {
      reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
          ->ensure_capacity(ckb_offset);
      self = reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)
                 ->get_at<tuple_ck>(root_ckb_offset);
      self->m_copy_el_offsets[i] = ckb_offset - root_ckb_offset;
      const char *field_arrmeta = src_arrmeta[0] + arrmeta_offsets[i];
      ckb_offset = self_af->instantiate(
          self_af, af_tp, NULL, ckb, ckb_offset, dst_tp, dst_arrmeta, nsrc,
          &field_types[i], &field_arrmeta, kernel_request_single, ectx,
          dynd::nd::array(), tp_vars);
    }
    return ckb_offset;
  }
  case pointer_type_id: {
    pydynd::nd::pointer_copy_kernel *self =
        pydynd::nd::pointer_copy_kernel::create(ckb, kernreq, ckb_offset);
    dynd::ndt::type src_value_tp =
        src_tp[0].extended<pointer_type>()->get_target_type();
    return self_af->instantiate(self_af, af_tp, NULL, ckb, ckb_offset, dst_tp,
                                dst_arrmeta, nsrc, &src_value_tp, src_arrmeta,
                                kernel_request_single, ectx, dynd::nd::array(),
                                tp_vars);
  }
  default:
    break;
  }

  if (src_tp[0].get_kind() == expr_kind) {
    dynd::nd::arrfunc af = dynd::nd::functional::chain(
        dynd::nd::copy, make_copy_to_pyobject_arrfunc(struct_as_pytuple),
        src_tp[0].value_type());
    return af.get()->instantiate(af.get(), af.get_type(), NULL, ckb, ckb_offset,
                                 dst_tp, dst_arrmeta, nsrc, src_tp, src_arrmeta,
                                 kernreq, ectx, kwds, tp_vars);
  }

  stringstream ss;
  ss << "Unable to copy dynd value with type " << src_tp[0]
     << " to a Python object";
  throw invalid_argument(ss.str());
}

static dynd::nd::arrfunc make_copy_to_pyobject_arrfunc(bool struct_as_pytuple)
{
  dynd::nd::array out_af = dynd::nd::empty("(Any) -> void");
  arrfunc_type_data *af =
      reinterpret_cast<arrfunc_type_data *>(out_af.get_readwrite_originptr());
  af->instantiate = &instantiate_copy_to_pyobject;
  *af->get_data_as<bool>() = struct_as_pytuple;
  out_af.flag_as_immutable();
  return out_af;
}

dynd::nd::arrfunc pydynd::nd::copy_to_pyobject_dict::make()
{
  PyDateTime_IMPORT;
  return make_copy_to_pyobject_arrfunc(false);
}

struct pydynd::nd::copy_to_pyobject_dict pydynd::nd::copy_to_pyobject_dict;

dynd::nd::arrfunc pydynd::nd::copy_to_pyobject_tuple::make()
{
  PyDateTime_IMPORT;
  return make_copy_to_pyobject_arrfunc(true);
}

struct pydynd::nd::copy_to_pyobject_tuple pydynd::nd::copy_to_pyobject_tuple;
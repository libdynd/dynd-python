//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// This header defines some wrapping functions to
// access various nd::array parameters
//

#ifndef _DYND__ARRAY_FUNCTIONS_HPP_
#define _DYND__ARRAY_FUNCTIONS_HPP_

#include <Python.h>

#include <sstream>

#include <dynd/array.hpp>
#include <dynd/array_range.hpp>
#include <dynd/json_parser.hpp>
#include <dynd/shape_tools.hpp>
#include <dynd/type_promotion.hpp>
#include <dynd/types/base_bytes_type.hpp>
#include <dynd/types/base_dim_type.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/struct_type.hpp>

#include "array_as_numpy.hpp"
#include "array_as_pep3118.hpp"
#include "array_from_py.hpp"
#include "array_conversions.hpp"
#include "utility_functions.hpp"
#include "type_functions.hpp"
#include "types/pyobject_type.hpp"
#include "visibility.hpp"

namespace pydynd {

inline dynd::nd::array make_strided_array(const dynd::ndt::type &dtp,
                                          intptr_t ndim, const intptr_t *shape)
{
  // Create the type of the result
  bool any_variable_dims = false;
  dynd::ndt::type array_tp =
      dynd::ndt::make_type(ndim, shape, dtp, any_variable_dims);

  // Determine the total data size
  size_t data_size;
  if (array_tp.is_builtin()) {
    data_size = array_tp.get_data_size();
  }
  else {
    data_size = array_tp.extended()->get_default_data_size();
  }

  dynd::intrusive_ptr<dynd::memory_block_data> result;
  char *data_ptr = NULL;
  if (array_tp.get_base_id() == dynd::memory_id) {
    result = dynd::make_array_memory_block(array_tp.get_arrmeta_size());
    array_tp.extended<dynd::ndt::base_memory_type>()->data_alloc(&data_ptr,
                                                                 data_size);
  }
  else {
    // Allocate the array arrmeta and data in one memory block
    result =
        dynd::make_array_memory_block(array_tp.get_arrmeta_size(), data_size,
                                      array_tp.get_data_alignment(), &data_ptr);
  }

  if (array_tp.get_flags() & dynd::type_flag_zeroinit) {
    if (array_tp.get_base_id() == dynd::memory_id) {
      array_tp.extended<dynd::ndt::base_memory_type>()->data_zeroinit(
          data_ptr, data_size);
    }
    else {
      memset(data_ptr, 0, data_size);
    }
  }

  // Fill in the preamble arrmeta
  dynd::array_preamble *ndo =
      reinterpret_cast<dynd::array_preamble *>(result.get());
  ndo->tp = array_tp;
  ndo->data = data_ptr;
  ndo->owner = NULL;
  ndo->flags = dynd::nd::read_access_flag | dynd::nd::write_access_flag;

  if (!any_variable_dims) {
    // Fill in the array arrmeta with strides and sizes
    dynd::fixed_dim_type_arrmeta *meta =
        reinterpret_cast<dynd::fixed_dim_type_arrmeta *>(ndo + 1);
    // Use the default construction to handle the uniform_tp's arrmeta
    intptr_t stride = dtp.get_data_size();
    if (stride == 0) {
      stride = dtp.extended()->get_default_data_size();
    }
    if (!dtp.is_builtin()) {
      dtp.extended()->arrmeta_default_construct(
          reinterpret_cast<char *>(meta + ndim), true);
    }
    for (intptr_t i = ndim - 1; i >= 0; --i) {
      intptr_t dim_size = shape[i];
      meta[i].stride = dim_size > 1 ? stride : 0;
      meta[i].dim_size = dim_size;
      stride *= dim_size;
    }
  }
  else {
    // Fill in the array arrmeta with strides and sizes
    char *meta = reinterpret_cast<char *>(ndo + 1);
    ndo->tp->arrmeta_default_construct(meta, true);
  }

  return dynd::nd::array(ndo, true);
}

inline dynd::nd::array pyobject_array()
{
  return dynd::nd::empty(dynd::ndt::make_type<pyobject_type>());
}

inline dynd::nd::array pyobject_array(PyObject *obj)
{
  dynd::nd::array a = dynd::nd::empty(dynd::ndt::make_type<pyobject_type>());
  *reinterpret_cast<PyObject **>(a.data()) = obj;

  return a;
}

PYDYND_API dynd::nd::array array_full(const dynd::ndt::type &d, PyObject *value,
                                      PyObject *access);
PYDYND_API dynd::nd::array array_full(PyObject *shape, const dynd::ndt::type &d,
                                      PyObject *value, PyObject *access);

inline bool array_is_c_contiguous(const dynd::nd::array &n)
{
  intptr_t ndim = n.get_ndim();
  dynd::dimvector shape(ndim), strides(ndim);
  n.get_shape(shape.get());
  n.get_strides(strides.get());
  return dynd::strides_are_c_contiguous(ndim, n.get_dtype().get_data_size(),
                                        shape.get(), strides.get());
}

inline bool array_is_f_contiguous(const dynd::nd::array &n)
{
  intptr_t ndim = n.get_ndim();
  dynd::dimvector shape(ndim), strides(ndim);
  n.get_shape(shape.get());
  n.get_strides(strides.get());
  return dynd::strides_are_f_contiguous(ndim, n.get_dtype().get_data_size(),
                                        shape.get(), strides.get());
}

inline std::string array_repr(const dynd::nd::array &n)
{
  std::stringstream n_ss;
  n_ss << n;
  std::stringstream ss;
  ss << "nd.";
  dynd::print_indented(ss, "   ", n_ss.str(), true);
  return ss.str();
}

inline PyObject *array_nonzero(const dynd::nd::array &n)
{
  // Implements the nonzero/conversion to boolean slot
  switch (n.get_type().value_type().get_base_id()) {
  case dynd::bool_kind_id:
  case dynd::uint_kind_id:
  case dynd::int_kind_id:
  case dynd::float_kind_id:
  case dynd::complex_kind_id:
    // Follow Python in not raising errors here
    if (n.as<bool>(dynd::assign_error_nocheck)) {
      Py_INCREF(Py_True);
      return Py_True;
    }
    else {
      Py_INCREF(Py_False);
      return Py_False;
    }
  case dynd::string_kind_id: {
    // Follow Python, return True if the string is nonempty, False otherwise
    dynd::nd::array n_eval = n.eval();
    const dynd::ndt::base_string_type *bsd =
        n_eval.get_type().extended<dynd::ndt::base_string_type>();
    const char *begin = NULL, *end = NULL;
    bsd->get_string_range(&begin, &end, n_eval.get()->metadata(),
                          n_eval.cdata());
    if (begin != end) {
      Py_INCREF(Py_True);
      return Py_True;
    }
    else {
      Py_INCREF(Py_False);
      return Py_False;
    }
  }
  case dynd::bytes_kind_id: {
    // Return True if there is a non-zero byte, False otherwise
    dynd::nd::array n_eval = n.eval();
    const dynd::ndt::base_bytes_type *bbd =
        n_eval.get_type().extended<dynd::ndt::base_bytes_type>();
    const char *begin = NULL, *end = NULL;
    bbd->get_bytes_range(&begin, &end, n_eval.get()->metadata(),
                         n_eval.cdata());
    while (begin != end) {
      if (*begin != 0) {
        Py_INCREF(Py_True);
        return Py_True;
      }
      else {
        ++begin;
      }
    }
    Py_INCREF(Py_False);
    return Py_False;
  }
  default:
    // TODO: Implement nd.any and nd.all, mention them
    //       here like NumPy does.
    PyErr_SetString(PyExc_ValueError, "the truth value of a dynd array with "
                                      "non-scalar type is ambiguous");
    throw std::exception();
  }
}

inline PyObject *array_get_shape(const dynd::nd::array &n)
{
  if (n.is_null()) {
    PyErr_SetString(PyExc_AttributeError,
                    "Cannot access attribute of null dynd array");
    throw std::exception();
  }
  size_t ndim = n.get_type().get_ndim();
  dynd::dimvector result(ndim);
  n.get_shape(result.get());
  return intptr_array_as_tuple(ndim, result.get());
}

inline PyObject *array_get_strides(const dynd::nd::array &n)
{
  if (n.is_null()) {
    PyErr_SetString(PyExc_AttributeError,
                    "Cannot access attribute of null dynd array");
    throw std::exception();
  }
  size_t ndim = n.get_type().get_ndim();
  dynd::dimvector result(ndim);
  n.get_strides(result.get());
  return intptr_array_as_tuple(ndim, result.get());
}

inline void
pyobject_as_irange_array(intptr_t &out_size,
                         dynd::shortvector<dynd::irange> &out_indices,
                         PyObject *subscript)
{
  if (!PyTuple_Check(subscript)) {
    // A single subscript
    out_size = 1;
    out_indices.init(1);
    out_indices[0] = pyobject_as_irange(subscript);
  }
  else {
    out_size = PyTuple_GET_SIZE(subscript);
    // Tuple of subscripts
    out_indices.init(out_size);
    for (Py_ssize_t i = 0; i < out_size; ++i) {
      out_indices[i] = pyobject_as_irange(PyTuple_GET_ITEM(subscript, i));
    }
  }
}

/**
 * Implementation of __getitem__ for the wrapped array object.
 */
inline dynd::nd::array array_getitem(const dynd::nd::array &n,
                                     PyObject *subscript)
{
  if (subscript == Py_Ellipsis) {
    return n.at_array(0, NULL);
  }
  else {
    // Convert the pyobject into an array of iranges
    intptr_t size;
    dynd::shortvector<dynd::irange> indices;
    pyobject_as_irange_array(size, indices, subscript);

    // Do an indexing operation
    return n.at_array(size, indices.get());
  }
}

/**
 * Implementation of __setitem__ for the wrapped dynd array object.
 */
inline void array_setitem(const dynd::nd::array &n, PyObject *subscript,
                          PyObject *value)
{
  if (subscript == Py_Ellipsis) {
    n.assign(value);
#if PY_VERSION_HEX < 0x03000000
  }
  else if (PyInt_Check(subscript)) {
    long i = PyInt_AS_LONG(subscript);
    n(i).assign(value);
#endif // PY_VERSION_HEX < 0x03000000
  }
  else if (PyLong_Check(subscript)) {
    intptr_t i = PyLong_AsSsize_t(subscript);
    if (i == -1 && PyErr_Occurred()) {
      throw std::runtime_error("error converting int value");
    }
    n(i).assign(value);
  }
  else {
    intptr_t size;
    dynd::shortvector<dynd::irange> indices;
    pyobject_as_irange_array(size, indices, subscript);
    n.at_array(size, indices.get(), false).assign(value);
  }
}

/**
 * Implementation of nd.range().
 */
inline dynd::nd::array array_range(PyObject *start, PyObject *stop,
                                   PyObject *step, PyObject *dt)
{
  dynd::nd::array start_nd, stop_nd, step_nd;
  dynd::ndt::type dt_nd;

  if (start != Py_None) {
    start_nd = array_from_py(start, 0, false);
  }
  else {
    start_nd = 0;
  }
  stop_nd = array_from_py(stop, 0, false);
  if (step != Py_None) {
    step_nd = array_from_py(step, 0, false);
  }
  else {
    step_nd = 1;
  }

  if (dt != Py_None) {
    dt_nd = dynd_ndt_as_cpp_type(dt);
  }
  else {
    dt_nd = promote_types_arithmetic(
        start_nd.get_type(),
        promote_types_arithmetic(stop_nd.get_type(), step_nd.get_type()));
  }

  start_nd = dynd::nd::empty(dt_nd).assign(start_nd);
  stop_nd = dynd::nd::empty(dt_nd).assign(stop_nd);
  step_nd = dynd::nd::empty(dt_nd).assign(step_nd);

  if (!start_nd.is_scalar() || !stop_nd.is_scalar() || !step_nd.is_scalar()) {
    throw std::runtime_error(
        "nd::range should only be called with scalar parameters");
  }

  return dynd::nd::range(dt_nd, start_nd.cdata(), stop_nd.cdata(),
                         step_nd.cdata());
}

/**
 * Implementation of nd.linspace().
 */
inline dynd::nd::array array_linspace(PyObject *start, PyObject *stop,
                                      PyObject *count, PyObject *dt)
{
  dynd::nd::array start_nd, stop_nd;
  intptr_t count_val = pyobject_as_index(count);
  start_nd = array_from_py(start, 0, false);
  stop_nd = array_from_py(stop, 0, false);
  if (dt == Py_None) {
    return dynd::nd::linspace(start_nd, stop_nd, count_val);
  }
  else {
    return dynd::nd::linspace(start_nd, stop_nd, count_val,
                              dynd_ndt_as_cpp_type(dt));
  }
}

/**
 * Implementation of nd.fields().
 */
inline dynd::nd::array nd_fields(const dynd::nd::array &n, PyObject *field_list)
{
  std::vector<std::string> selected_fields;
  pyobject_as_vector_string(field_list, selected_fields);

  // TODO: Move this implementation into dynd
  dynd::ndt::type fdt = n.get_dtype();
  if (fdt.get_id() != dynd::struct_id) {
    std::stringstream ss;
    ss << "nd.fields must be given a dynd array of 'struct' kind, not ";
    ss << fdt;
    throw std::runtime_error(ss.str());
  }
  const dynd::ndt::struct_type *bsd = fdt.extended<dynd::ndt::struct_type>();

  if (selected_fields.empty()) {
    throw std::runtime_error(
        "nd.fields requires at least one field name to be specified");
  }
  // Construct the field mapping and output field types
  std::vector<intptr_t> selected_index(selected_fields.size());
  std::vector<dynd::ndt::type> selected__types(selected_fields.size());
  for (size_t i = 0; i != selected_fields.size(); ++i) {
    selected_index[i] = bsd->get_field_index(selected_fields[i]);
    if (selected_index[i] < 0) {
      std::stringstream ss;
      ss << "field name ";
      dynd::print_escaped_utf8_string(ss, selected_fields[i]);
      ss << " does not exist in dynd type " << fdt;
      throw std::runtime_error(ss.str());
    }
    selected__types[i] = bsd->get_field_type(selected_index[i]);
  }
  // Create the result udt
  dynd::ndt::type rudt =
      dynd::ndt::struct_type::make(selected_fields, selected__types);
  dynd::ndt::type result_tp = n.get_type().with_replaced_dtype(rudt);
  const dynd::ndt::struct_type *rudt_bsd =
      rudt.extended<dynd::ndt::struct_type>();

  // Allocate the new memory block.
  size_t arrmeta_size = result_tp.get_arrmeta_size();
  dynd::nd::array result(reinterpret_cast<dynd::array_preamble *>(
                             dynd::make_array_memory_block(arrmeta_size).get()),
                         true);

  // Clone the data pointer
  result.get()->data = n.get()->data;
  result.get()->owner = n.get()->owner;
  if (!result.get()->owner) {
    result.get()->owner = n.get();
  }

  // Copy the flags
  result.get()->flags = n.get()->flags;

  // Set the type and transform the arrmeta
  result.get()->tp = result_tp;
  // First copy all the array data type arrmeta
  dynd::ndt::type tmp_dt = result_tp;
  char *dst_arrmeta = result.get()->metadata();
  const char *src_arrmeta = n.get()->metadata();
  while (tmp_dt.get_ndim() > 0) {
    if (tmp_dt.get_base_id() != dynd::dim_kind_id) {
      throw std::runtime_error(
          "nd.fields doesn't support dimensions with pointers yet");
    }
    const dynd::ndt::base_dim_type *budd =
        tmp_dt.extended<dynd::ndt::base_dim_type>();
    size_t offset = budd->arrmeta_copy_construct_onedim(
        dst_arrmeta, src_arrmeta,
        dynd::intrusive_ptr<dynd::memory_block_data>(n.get(), true));
    dst_arrmeta += offset;
    src_arrmeta += offset;
    tmp_dt = budd->get_element_type();
  }
  // Then create the arrmeta for the new struct
  const size_t *arrmeta_offsets = bsd->get_arrmeta_offsets_raw();
  const size_t *result_arrmeta_offsets = rudt_bsd->get_arrmeta_offsets_raw();
  const size_t *data_offsets = bsd->get_data_offsets(src_arrmeta);
  size_t *result_data_offsets = reinterpret_cast<size_t *>(dst_arrmeta);
  for (size_t i = 0; i != selected_fields.size(); ++i) {
    const dynd::ndt::type &dt = selected__types[i];
    // Copy the data offset
    result_data_offsets[i] = data_offsets[selected_index[i]];
    // Copy the arrmeta for this field
    if (dt.get_arrmeta_size() > 0) {
      dt.extended()->arrmeta_copy_construct(
          dst_arrmeta + result_arrmeta_offsets[i],
          src_arrmeta + arrmeta_offsets[selected_index[i]],
          dynd::intrusive_ptr<dynd::memory_block_data>(n.get(), true));
    }
  }

  return result;
}

inline const char *array_access_flags_string(const dynd::nd::array &n)
{
  if (n.is_null()) {
    PyErr_SetString(PyExc_AttributeError,
                    "Cannot access attribute of null dynd array");
    throw std::exception();
  }
  switch (n.get_access_flags()) {
  case dynd::nd::read_access_flag | dynd::nd::immutable_access_flag:
    return "immutable";
  case dynd::nd::read_access_flag:
    return "readonly";
  case dynd::nd::read_access_flag | dynd::nd::write_access_flag:
    return "readwrite";
  default:
    return "<invalid flags>";
  }
}

inline dynd::nd::array dynd_parse_json_type(const dynd::ndt::type &tp,
                                            const dynd::nd::array &json,
                                            PyObject *ectx_obj)
{
  return dynd::parse_json(tp, json, &dynd::eval::default_eval_context);
}

inline void dynd_parse_json_array(dynd::nd::array &out,
                                  const dynd::nd::array &json,
                                  PyObject *ectx_obj)
{
  dynd::parse_json(out, json, &dynd::eval::default_eval_context);
}

} // namespace pydynd

#endif // _DYND__ARRAY_FUNCTIONS_HPP_

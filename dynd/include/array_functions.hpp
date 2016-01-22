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
#include <dynd/shape_tools.hpp>
#include <dynd/json_parser.hpp>
#include <dynd/types/base_bytes_type.hpp>
#include <dynd/types/string_type.hpp>

#include "visibility.hpp"
#include "array_from_py.hpp"
#include "array_as_numpy.hpp"
#include "array_as_pep3118.hpp"
#include "utility_functions.hpp"

#include "wrapper.hpp"

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
  if (array_tp.get_kind() == dynd::memory_kind) {
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
    if (array_tp.get_kind() == dynd::memory_kind) {
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

PYDYND_API dynd::nd::array pyobject_array(PyObject *obj);

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
  switch (n.get_type().value_type().get_kind()) {
  case dynd::bool_kind:
  case dynd::uint_kind:
  case dynd::sint_kind:
  case dynd::real_kind:
  case dynd::complex_kind:
    // Follow Python in not raising errors here
    if (n.as<bool>(dynd::assign_error_nocheck)) {
      Py_INCREF(Py_True);
      return Py_True;
    }
    else {
      Py_INCREF(Py_False);
      return Py_False;
    }
  case dynd::string_kind: {
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
  case dynd::bytes_kind: {
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
  case dynd::datetime_kind: {
    // Dates and datetimes are never zero
    // TODO: What to do with NA value?
    Py_INCREF(Py_True);
    return Py_True;
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

inline void pyobject_as_irange_array(intptr_t &out_size,
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
PYDYND_API dynd::nd::array array_getitem(const dynd::nd::array &n,
                                         PyObject *subscript);

/**
 * Implementation of __setitem__ for the wrapped dynd array object.
 */
PYDYND_API void array_setitem(const dynd::nd::array &n, PyObject *subscript,
                              PyObject *value);

/**
 * Implementation of nd.range().
 */
PYDYND_API dynd::nd::array array_range(PyObject *start, PyObject *stop,
                                       PyObject *step, PyObject *dt);

/**
 * Implementation of nd.linspace().
 */
PYDYND_API dynd::nd::array array_linspace(PyObject *start, PyObject *stop,
                                          PyObject *count, PyObject *dt);

/**
 * Implementation of nd.fields().
 */
PYDYND_API dynd::nd::array nd_fields(const dynd::nd::array &n,
                                     PyObject *field_list);

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

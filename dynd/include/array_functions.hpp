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

#include "visibility.hpp"
#include "array_from_py.hpp"
#include "array_as_py.hpp"
#include "array_as_numpy.hpp"
#include "array_as_pep3118.hpp"
#include "eval_context_functions.hpp"

#include "wrapper.hpp"

namespace dynd {
namespace nd {

  inline nd::array make_strided_array(const ndt::type &dtp, intptr_t ndim,
                                      const intptr_t *shape)
  {
    // Create the type of the result
    bool any_variable_dims = false;
    ndt::type array_tp = ndt::make_type(ndim, shape, dtp, any_variable_dims);

    // Determine the total data size
    size_t data_size;
    if (array_tp.is_builtin()) {
      data_size = array_tp.get_data_size();
    }
    else {
      data_size = array_tp.extended()->get_default_data_size();
    }

    intrusive_ptr<memory_block_data> result;
    char *data_ptr = NULL;
    if (array_tp.get_kind() == memory_kind) {
      result = make_array_memory_block(array_tp.get_arrmeta_size());
      array_tp.extended<ndt::base_memory_type>()->data_alloc(&data_ptr,
                                                             data_size);
    }
    else {
      // Allocate the array arrmeta and data in one memory block
      result =
          make_array_memory_block(array_tp.get_arrmeta_size(), data_size,
                                  array_tp.get_data_alignment(), &data_ptr);
    }

    if (array_tp.get_flags() & type_flag_zeroinit) {
      if (array_tp.get_kind() == memory_kind) {
        array_tp.extended<ndt::base_memory_type>()->data_zeroinit(data_ptr,
                                                                  data_size);
      }
      else {
        memset(data_ptr, 0, data_size);
      }
    }

    // Fill in the preamble arrmeta
    array_preamble *ndo = reinterpret_cast<array_preamble *>(result.get());
    ndo->tp = array_tp;
    ndo->data = data_ptr;
    ndo->owner = NULL;
    ndo->flags = read_access_flag | write_access_flag;

    if (!any_variable_dims) {
      // Fill in the array arrmeta with strides and sizes
      fixed_dim_type_arrmeta *meta =
          reinterpret_cast<fixed_dim_type_arrmeta *>(ndo + 1);
      // Use the default construction to handle the uniform_tp's arrmeta
      intptr_t stride = dtp.get_data_size();
      if (stride == 0) {
        stride = dtp.extended()->get_default_data_size();
      }
      if (!dtp.is_builtin()) {
        dtp.extended()->arrmeta_default_construct(
            reinterpret_cast<char *>(meta + ndim), true);
      }
      for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; --i) {
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

    return array(ndo, true);
  }
}
}

namespace pydynd {

PYDYND_API void array_init_from_pyobject(dynd::nd::array &n, PyObject *obj,
                                         PyObject *dt, bool uniform,
                                         PyObject *access);
PYDYND_API void array_init_from_pyobject(dynd::nd::array &n, PyObject *obj,
                                         PyObject *access);

PYDYND_API dynd::nd::array array_asarray(PyObject *obj, PyObject *access);

PYDYND_API dynd::nd::array array_eval(const dynd::nd::array &n, PyObject *ectx);

PYDYND_API dynd::nd::array array_zeros(const dynd::ndt::type &d,
                                       PyObject *access);
PYDYND_API dynd::nd::array
array_zeros(PyObject *shape, const dynd::ndt::type &d, PyObject *access);

PYDYND_API dynd::nd::array array_ones(const dynd::ndt::type &d,
                                      PyObject *access);
PYDYND_API dynd::nd::array array_ones(PyObject *shape, const dynd::ndt::type &d,
                                      PyObject *access);

PYDYND_API dynd::nd::array array_full(const dynd::ndt::type &d, PyObject *value,
                                      PyObject *access);
PYDYND_API dynd::nd::array array_full(PyObject *shape, const dynd::ndt::type &d,
                                      PyObject *value, PyObject *access);

PYDYND_API dynd::nd::array array_empty(const dynd::ndt::type &d,
                                       PyObject *access);
PYDYND_API dynd::nd::array
array_empty(PyObject *shape, const dynd::ndt::type &d, PyObject *access);

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

PYDYND_API PyObject *array_index(const dynd::nd::array &n);
PYDYND_API PyObject *array_nonzero(const dynd::nd::array &n);
PYDYND_API PyObject *array_int(const dynd::nd::array &n);
PYDYND_API PyObject *array_float(const dynd::nd::array &n);
PYDYND_API PyObject *array_complex(const dynd::nd::array &n);

PYDYND_API dynd::nd::array array_cast(const dynd::nd::array &n,
                                      const dynd::ndt::type &dt);

PYDYND_API dynd::nd::array array_ucast(const dynd::nd::array &n,
                                       const dynd::ndt::type &dt,
                                       intptr_t replace_ndim);

PYDYND_API PyObject *array_get_shape(const dynd::nd::array &n);

PYDYND_API PyObject *array_get_strides(const dynd::nd::array &n);

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
  return dynd::parse_json(tp, json, eval_context_from_pyobj(ectx_obj));
}

inline void dynd_parse_json_array(dynd::nd::array &out,
                                  const dynd::nd::array &json,
                                  PyObject *ectx_obj)
{
  dynd::parse_json(out, json, eval_context_from_pyobj(ectx_obj));
}

} // namespace pydynd

#endif // _DYND__ARRAY_FUNCTIONS_HPP_

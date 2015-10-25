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

#include "config.hpp"
#include "array_from_py.hpp"
#include "array_as_py.hpp"
#include "array_as_numpy.hpp"
#include "array_as_pep3118.hpp"
#include "eval_context_functions.hpp"

#include "wrapper.hpp"

typedef DyND_PyWrapperObject<dynd::nd::array> DyND_PyArrayObject;

inline int DyND_PyArray_Check(PyObject *obj)
{
  return DyND_PyWrapper_Check<dynd::nd::array>(obj);
}

inline int DyND_PyArray_CheckExact(PyObject *obj)
{
  return DyND_PyWrapper_CheckExact<dynd::nd::array>(obj);
}

namespace pydynd {

PYDYND_API void array_init_from_pyobject(dynd::nd::array &n, PyObject *obj,
                                         PyObject *dt, bool uniform,
                                         PyObject *access);
PYDYND_API void array_init_from_pyobject(dynd::nd::array &n, PyObject *obj,
                                         PyObject *access);

PYDYND_API dynd::nd::array array_view(PyObject *obj, PyObject *type,
                                      PyObject *access);
PYDYND_API dynd::nd::array array_asarray(PyObject *obj, PyObject *access);

PYDYND_API dynd::nd::array array_eval(const dynd::nd::array &n, PyObject *ectx);
dynd::nd::array array_eval_copy(const dynd::nd::array &n, PyObject *access,
                                PyObject *ectx);

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

inline dynd::nd::array array_empty_like(const dynd::nd::array &n)
{
  return dynd::nd::empty_like(n);
}

inline dynd::nd::array array_empty_like(const dynd::nd::array &n,
                                        const dynd::ndt::type &d)
{
  return dynd::nd::empty_like(n, d);
}

PYDYND_API dynd::nd::array array_memmap(PyObject *filename, PyObject *begin,
                                        PyObject *end, PyObject *access);

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

inline dynd::nd::array array_add(const dynd::nd::array &lhs,
                                 const dynd::nd::array &rhs)
{
  return lhs + rhs;
}

inline dynd::nd::array array_subtract(const dynd::nd::array &lhs,
                                      const dynd::nd::array &rhs)
{
  return lhs - rhs;
}

inline dynd::nd::array array_multiply(const dynd::nd::array &lhs,
                                      const dynd::nd::array &rhs)
{
  return lhs * rhs;
}

inline dynd::nd::array array_divide(const dynd::nd::array &lhs,
                                    const dynd::nd::array &rhs)
{
  return lhs / rhs;
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

PYDYND_API PyObject *array_str(const dynd::nd::array &n);
PYDYND_API PyObject *array_unicode(const dynd::nd::array &n);
PYDYND_API PyObject *array_index(const dynd::nd::array &n);
PYDYND_API PyObject *array_nonzero(const dynd::nd::array &n);
PYDYND_API PyObject *array_int(const dynd::nd::array &n);
PYDYND_API PyObject *array_float(const dynd::nd::array &n);
PYDYND_API PyObject *array_complex(const dynd::nd::array &n);

inline std::string array_debug_print(const dynd::nd::array &n)
{
  std::stringstream ss;
  n.debug_print(ss);
  return ss.str();
}

PYDYND_API dynd::nd::array array_cast(const dynd::nd::array &n,
                                      const dynd::ndt::type &dt);

PYDYND_API dynd::nd::array array_ucast(const dynd::nd::array &n,
                                       const dynd::ndt::type &dt,
                                       intptr_t replace_ndim);

PyObject *array_adapt(PyObject *a, PyObject *tp_obj, PyObject *adapt_op);

PYDYND_API PyObject *array_get_shape(const dynd::nd::array &n);

PYDYND_API PyObject *array_get_strides(const dynd::nd::array &n);

bool array_is_scalar(const dynd::nd::array &n);

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

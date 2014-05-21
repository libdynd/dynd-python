//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
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

#include "array_from_py.hpp"
#include "array_as_py.hpp"
#include "array_as_numpy.hpp"
#include "array_as_pep3118.hpp"
#include "placement_wrappers.hpp"
#include "eval_context_functions.hpp"

namespace pydynd {

/**
 * This is the typeobject and struct of w_array from Cython.
 */
extern PyTypeObject *WArray_Type;
inline bool WArray_CheckExact(PyObject *obj) {
    return Py_TYPE(obj) == WArray_Type;
}
inline bool WArray_Check(PyObject *obj) {
    return PyObject_TypeCheck(obj, WArray_Type);
}
struct WArray {
  PyObject_HEAD;
  // This is array_placement_wrapper in Cython-land
  dynd::nd::array v;
};
void init_w_array_typeobject(PyObject *type);

PyObject *wrap_array(const dynd::nd::array& n);
PyObject *wrap_array(const dynd::nd::arrfunc& n);


void array_init_from_pyobject(dynd::nd::array& n, PyObject* obj, PyObject *dt, bool uniform, PyObject *access);
void array_init_from_pyobject(dynd::nd::array& n, PyObject* obj, PyObject *access);

dynd::nd::array array_view(PyObject *obj, PyObject *type, PyObject *access);
dynd::nd::array array_asarray(PyObject *obj, PyObject *access);

dynd::nd::array array_eval(const dynd::nd::array& n, PyObject *ectx);
dynd::nd::array array_eval_copy(const dynd::nd::array& n,
                PyObject* access,
                PyObject *ectx);

dynd::nd::array array_zeros(const dynd::ndt::type& d, PyObject *access);
dynd::nd::array array_zeros(PyObject *shape, const dynd::ndt::type& d, PyObject *access);

dynd::nd::array array_ones(const dynd::ndt::type& d, PyObject *access);
dynd::nd::array array_ones(PyObject *shape, const dynd::ndt::type& d, PyObject *access);

dynd::nd::array array_full(const dynd::ndt::type& d, PyObject *value, PyObject *access);
dynd::nd::array array_full(PyObject *shape, const dynd::ndt::type& d, PyObject *value, PyObject *access);

dynd::nd::array array_empty(const dynd::ndt::type& d, PyObject *access);
dynd::nd::array array_empty(PyObject *shape, const dynd::ndt::type& d, PyObject *access);

inline dynd::nd::array array_empty_like(const dynd::nd::array& n)
{
    return dynd::nd::empty_like(n);
}

inline dynd::nd::array array_empty_like(const dynd::nd::array& n, const dynd::ndt::type& d)
{
    return dynd::nd::empty_like(n, d);
}

dynd::nd::array array_memmap(PyObject *filename, PyObject *begin, PyObject *end, PyObject *access);

inline bool array_is_c_contiguous(const dynd::nd::array& n)
{
    intptr_t ndim = n.get_ndim();
    dynd::dimvector shape(ndim), strides(ndim);
    n.get_shape(shape.get());
    n.get_strides(strides.get());
    return dynd::strides_are_c_contiguous(ndim,
                    n.get_dtype().get_data_size(),
                    shape.get(), strides.get());
}

inline bool array_is_f_contiguous(const dynd::nd::array& n)
{
    intptr_t ndim = n.get_ndim();
    dynd::dimvector shape(ndim), strides(ndim);
    n.get_shape(shape.get());
    n.get_strides(strides.get());
    return dynd::strides_are_f_contiguous(ndim,
                    n.get_dtype().get_data_size(),
                    shape.get(), strides.get());
}

inline dynd::nd::array array_add(const dynd::nd::array& lhs, const dynd::nd::array& rhs)
{
    return lhs + rhs;
}

inline dynd::nd::array array_subtract(const dynd::nd::array& lhs, const dynd::nd::array& rhs)
{
    return lhs - rhs;
}

inline dynd::nd::array array_multiply(const dynd::nd::array& lhs, const dynd::nd::array& rhs)
{
    return lhs * rhs;
}

inline dynd::nd::array array_divide(const dynd::nd::array& lhs, const dynd::nd::array& rhs)
{
    return lhs / rhs;
}

inline std::string array_repr(const dynd::nd::array& n)
{
    std::stringstream ss;
    ss << "nd." << n;
    return ss.str();
}

PyObject *array_str(const dynd::nd::array& n);
PyObject *array_unicode(const dynd::nd::array& n);
PyObject *array_index(const dynd::nd::array& n);
PyObject *array_nonzero(const dynd::nd::array& n);
PyObject *array_int(const dynd::nd::array& n);
PyObject *array_float(const dynd::nd::array& n);
PyObject *array_complex(const dynd::nd::array& n);

inline std::string array_debug_print(const dynd::nd::array& n)
{
    std::stringstream ss;
    n.debug_print(ss);
    return ss.str();
}

bool array_contains(const dynd::nd::array& n, PyObject *x);


dynd::nd::array array_cast(const dynd::nd::array& n, const dynd::ndt::type& dt,
                PyObject *assign_error_obj);

dynd::nd::array array_ucast(const dynd::nd::array& n, const dynd::ndt::type& dt,
                intptr_t replace_ndim, PyObject *assign_error_obj);

PyObject *array_get_shape(const dynd::nd::array& n);

PyObject *array_get_strides(const dynd::nd::array& n);

/**
 * Implementation of __getitem__ for the wrapped array object.
 */
dynd::nd::array array_getitem(const dynd::nd::array& n, PyObject *subscript);

/**
 * Implementation of __setitem__ for the wrapped dynd array object.
 */
void array_setitem(const dynd::nd::array& n, PyObject *subscript, PyObject *value);

/**
 * Implementation of nd.range().
 */
dynd::nd::array array_range(PyObject *start, PyObject *stop, PyObject *step, PyObject *dt);

/**
 * Implementation of nd.linspace().
 */
dynd::nd::array array_linspace(PyObject *start, PyObject *stop, PyObject *count, PyObject *dt);

/**
 * Implementation of nd.fields().
 */
dynd::nd::array nd_fields(const dynd::nd::array& n, PyObject *field_list);

inline const char *array_access_flags_string(const dynd::nd::array& n) {
    switch (n.get_access_flags()) {
        case dynd::nd::read_access_flag|dynd::nd::immutable_access_flag:
            return "immutable";
        case dynd::nd::read_access_flag:
            return "readonly";
        case dynd::nd::read_access_flag|dynd::nd::write_access_flag:
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

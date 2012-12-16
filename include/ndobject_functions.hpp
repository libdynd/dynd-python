//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//
// This header defines some wrapping functions to
// access various ndobject parameters
//

#ifndef _DYND__NDARRAY_FUNCTIONS_HPP_
#define _DYND__NDARRAY_FUNCTIONS_HPP_

#include "Python.h"

#include <sstream>

#include <dynd/ndobject.hpp>

#include "ndobject_from_py.hpp"
#include "ndobject_as_py.hpp"
#include "ndobject_as_pep3118.hpp"
#include "placement_wrappers.hpp"

namespace pydynd {

/**
 * This is the typeobject and struct of w_ndobject from Cython.
 */
extern PyTypeObject *WNDObject_Type;
inline bool WNDObject_CheckExact(PyObject *obj) {
    return Py_TYPE(obj) == WNDObject_Type;
}
inline bool WNDObject_Check(PyObject *obj) {
    return PyObject_TypeCheck(obj, WNDObject_Type);
}
struct WNDObject {
  PyObject_HEAD;
  // This is ndobject_placement_wrapper in Cython-land
  dynd::ndobject v;
};
void init_w_ndobject_typeobject(PyObject *type);

inline PyObject *wrap_ndobject(const dynd::ndobject& n) {
    WNDObject *result = (WNDObject *)WNDObject_Type->tp_alloc(WNDObject_Type, 0);
    if (!result) {
        throw std::runtime_error("");
    }
    // Calling tp_alloc doesn't call Cython's __cinit__, so do the placement new here
    pydynd::placement_new(reinterpret_cast<pydynd::ndobject_placement_wrapper &>(result->v));
    result->v = n;
    return (PyObject *)result;
}

inline void ndobject_init_from_pyobject(dynd::ndobject& n, PyObject* obj)
{
    n = ndobject_from_py(obj);
}
dynd::ndobject ndobject_vals(const dynd::ndobject& n);
dynd::ndobject ndobject_eval_copy(const dynd::ndobject& n, PyObject* access_flags, const dynd::eval::eval_context *ectx = &dynd::eval::default_eval_context);

inline dynd::ndobject ndobject_empty_like(const dynd::ndobject& n)
{
    return dynd::empty_like(n);
}

inline dynd::ndobject ndobject_empty_like(const dynd::ndobject& n, const dynd::dtype& d)
{
    return dynd::empty_like(n, d);
}

inline dynd::ndobject ndobject_add(const dynd::ndobject& lhs, const dynd::ndobject& rhs)
{
    throw std::runtime_error("TODO: ndobject_add");
//    return lhs + rhs;
}

inline dynd::ndobject ndobject_subtract(const dynd::ndobject& lhs, const dynd::ndobject& rhs)
{
    throw std::runtime_error("TODO: ndobject_subtract");
//    return lhs - rhs;
}

inline dynd::ndobject ndobject_multiply(const dynd::ndobject& lhs, const dynd::ndobject& rhs)
{
    throw std::runtime_error("TODO: ndobject_multiply");
//    return lhs * rhs;
}

inline dynd::ndobject ndobject_divide(const dynd::ndobject& lhs, const dynd::ndobject& rhs)
{
    throw std::runtime_error("TODO: ndobject_divide");
//    return lhs / rhs;
}

inline std::string ndobject_str(const dynd::ndobject& n)
{
    std::stringstream ss;
    n.get_dtype().print_data(ss, n.get_ndo_meta(), n.get_readonly_originptr());
    return ss.str();
}

inline std::string ndobject_repr(const dynd::ndobject& n)
{
    std::stringstream ss;
    ss << "nd." << n;
    return ss.str();
}

inline std::string ndobject_debug_print(const dynd::ndobject& n)
{
    std::stringstream ss;
    n.debug_print(ss);
    return ss.str();
}

dynd::ndobject ndobject_cast_scalars(const dynd::ndobject& n, const dynd::dtype& dt, PyObject *assign_error_obj);
dynd::ndobject ndobject_cast_udtype(const dynd::ndobject& n, const dynd::dtype& dt, PyObject *assign_error_obj);

PyObject *ndobject_get_shape(const dynd::ndobject& n);

PyObject *ndobject_get_strides(const dynd::ndobject& n);

/**
 * Implementation of __getitem__ for the wrapped ndobject object.
 */
dynd::ndobject ndobject_getitem(const dynd::ndobject& n, PyObject *subscript);

/**
 * Implementation of nd.arange().
 */
dynd::ndobject ndobject_arange(PyObject *start, PyObject *stop, PyObject *step);

/**
 * Implementation of nd.linspace().
 */
dynd::ndobject ndobject_linspace(PyObject *start, PyObject *stop, PyObject *count);

/**
 * Implementation of nd.groupby().
 */
dynd::ndobject ndobject_groupby(const dynd::ndobject& data, const dynd::ndobject& by, const dynd::dtype& groups);

} // namespace pydynd

#endif // _DYND__NDARRAY_FUNCTIONS_HPP_

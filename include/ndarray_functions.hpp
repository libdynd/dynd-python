//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//
// This header defines some wrapping functions to
// access various ndarray parameters
//

#ifndef _DND__NDARRAY_FUNCTIONS_HPP_
#define _DND__NDARRAY_FUNCTIONS_HPP_

#include "Python.h"

#include <sstream>

#include <dnd/ndarray.hpp>

#include "ndarray_from_py.hpp"
#include "ndarray_as_py.hpp"

namespace pydynd {

/**
 * This is the typeobject and struct of w_ndarray from Cython.
 */
extern PyTypeObject *WNDArray_Type;
inline bool WNDArray_CheckExact(PyObject *obj) {
    return Py_TYPE(obj) == WNDArray_Type;
}
inline bool WNDArray_Check(PyObject *obj) {
    return PyObject_TypeCheck(obj, WNDArray_Type);
}
struct WNDArray {
  PyObject_HEAD;
  // This is ndarray_placement_wrapper in Cython-land
  dynd::ndarray v;
};
void init_w_ndarray_typeobject(PyObject *type);

inline void ndarray_init_from_pyobject(dynd::ndarray& n, PyObject* obj)
{
    n = ndarray_from_py(obj);
}
dynd::ndarray ndarray_vals(const dynd::ndarray& n);
dynd::ndarray ndarray_eval_copy(const dynd::ndarray& n, PyObject* access_flags, const dynd::eval::eval_context *ectx = &dynd::eval::default_eval_context);

inline dynd::ndarray ndarray_add(const dynd::ndarray& lhs, const dynd::ndarray& rhs)
{
    return lhs + rhs;
}

inline dynd::ndarray ndarray_subtract(const dynd::ndarray& lhs, const dynd::ndarray& rhs)
{
    return lhs - rhs;
}

inline dynd::ndarray ndarray_multiply(const dynd::ndarray& lhs, const dynd::ndarray& rhs)
{
    return lhs * rhs;
}

inline dynd::ndarray ndarray_divide(const dynd::ndarray& lhs, const dynd::ndarray& rhs)
{
    return lhs / rhs;
}

inline std::string ndarray_str(const dynd::ndarray& n)
{
    std::stringstream ss;
    ss << n;
    return ss.str();
}

inline std::string ndarray_repr(const dynd::ndarray& n)
{
    std::stringstream ss;
    ss << "nd." << n;
    return ss.str();
}

inline std::string ndarray_debug_dump(const dynd::ndarray& n)
{
    std::stringstream ss;
    n.debug_dump(ss);
    return ss.str();
}

dynd::ndarray ndarray_as_dtype(const dynd::ndarray& n, const dynd::dtype& dt, PyObject *assign_error_obj);

/**
 * Implementation of __getitem__ for the wrapped ndarray object.
 */
dynd::ndarray ndarray_getitem(const dynd::ndarray& n, PyObject *subscript);

/**
 * Implementation of nd.arange().
 */
dynd::ndarray ndarray_arange(PyObject *start, PyObject *stop, PyObject *step);

/**
 * Implementation of nd.linspace().
 */
dynd::ndarray ndarray_linspace(PyObject *start, PyObject *stop, PyObject *count);

/**
 * Implementation of nd.groupby().
 */
dynd::ndarray ndarray_groupby(const dynd::ndarray& data, const dynd::ndarray& by, const dynd::dtype& groups);

} // namespace pydynd

#endif // _DND__NDARRAY_FUNCTIONS_HPP_

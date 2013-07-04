//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__GFUNC_CALLABLE_FUNCTIONS_HPP_
#define _DYND__GFUNC_CALLABLE_FUNCTIONS_HPP_

#include "Python.h"

#include <dynd/dtype.hpp>
#include <dynd/gfunc/callable.hpp>

#include <sstream>

namespace pydynd {

struct array_callable_wrapper {
    dynd::nd::array n;
    dynd::gfunc::callable c;
    std::string funcname;
};

/**
 * This is the typeobject and struct of w_array_callable from Cython.
 */
extern PyTypeObject *WArrayCallable_Type;
inline bool WArrayCallable_CheckExact(PyObject *obj) {
    return Py_TYPE(obj) == WArrayCallable_Type;
}
inline bool WArrayCallable_Check(PyObject *obj) {
    return PyObject_TypeCheck(obj, WArrayCallable_Type);
}
struct WArrayCallable {
  PyObject_HEAD;
  // This is array_placement_wrapper in Cython-land
  array_callable_wrapper v;
};
void init_w_array_callable_typeobject(PyObject *type);

/**
 * This calls the callable in the array_callable_wrapper, which was
 * returned as a property of an array.
 */
PyObject *array_callable_call(const array_callable_wrapper& ncw, PyObject *args, PyObject *kwargs);

struct dtype_callable_wrapper {
    dynd::dtype d;
    dynd::gfunc::callable c;
    std::string funcname;
};

/**
 * This is the typeobject and struct of w_dtype_callable from Cython.
 */
extern PyTypeObject *WDTypeCallable_Type;
inline bool WDTypeCallable_CheckExact(PyObject *obj) {
    return Py_TYPE(obj) == WDTypeCallable_Type;
}
inline bool WDTypeCallable_Check(PyObject *obj) {
    return PyObject_TypeCheck(obj, WDTypeCallable_Type);
}
struct WDTypeCallable {
  PyObject_HEAD;
  // This is dtype_placement_wrapper in Cython-land
  dtype_callable_wrapper v;
};
void init_w_dtype_callable_typeobject(PyObject *type);
/**
 * This calls the callable in the dtype_callable_wrapper, which was
 * returned as a property of a dtype.
 */
PyObject *dtype_callable_call(const dtype_callable_wrapper& ccw, PyObject *args, PyObject *kwargs);

/**
 * This calls the dynamic __construct__ function attached to the dtype.
 */
PyObject *call_dtype_constructor_function(const dynd::dtype& dt, PyObject *args, PyObject *kwargs);

/**
 * Adds all the dynamic names exposed by the dtype to the provided dict.
 */
void add_dtype_names_to_dir_dict(const dynd::dtype& dt, PyObject *dict);
/**
 * Retrieves a dynamic property from the dtype as a Python object.
 */
PyObject *get_dtype_dynamic_property(const dynd::dtype& dt, PyObject *name);

/**
 * Adds all the dynamic names exposed by the array to the provided dict.
 */
void add_array_names_to_dir_dict(const dynd::nd::array& n, PyObject *dict);
/**
 * Retrieves a dynamic property from the nd::array.
 */
PyObject *get_array_dynamic_property(const dynd::nd::array& n, PyObject *name);
/**
 * Sets a dynamic property of the nd::array.
 */
void set_array_dynamic_property(const dynd::nd::array& n, PyObject *name, PyObject *value);

/**
 * Calls the callable with the single dtype parameter
 *
 * \param funcname  The callable name.
 * \param c  The callable.
 * \param dt  The first parameter for the callable.
 */
PyObject *call_gfunc_callable(const std::string& funcname, const dynd::gfunc::callable& c, const dynd::dtype& dt);

/**
 * Calls the callable with the single nd::array parameter
 *
 * \param funcname  The callable name.
 * \param c  The callable.
 * \param n  The first parameter for the callable.
 */
dynd::nd::array call_gfunc_callable(const std::string& funcname, const dynd::gfunc::callable& c, const dynd::nd::array& n);

/**
 * Returns a wrapper for the callable with the nd::array as the first parameter.
 *
 * \param funcname  The callable name.
 * \param c  The callable.
 * \param n  The first parameter for the callable.
 */
PyObject *wrap_array_callable(const std::string& funcname, const dynd::gfunc::callable& c, const dynd::nd::array& n);

/**
 * Returns a wrapper for the callable with the dtype as the first parameter.
 *
 * \param funcname  The callable name.
 * \param c  The callable.
 * \param d  The first parameter for the callable.
 */
PyObject *wrap_dtype_callable(const std::string& funcname, const dynd::gfunc::callable& c, const dynd::dtype& d);

} // namespace pydynd

#endif // _DYND__GFUNC_CALLABLE_FUNCTIONS_HPP_


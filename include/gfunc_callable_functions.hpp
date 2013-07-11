//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__GFUNC_CALLABLE_FUNCTIONS_HPP_
#define _DYND__GFUNC_CALLABLE_FUNCTIONS_HPP_

#include "Python.h"

#include <dynd/type.hpp>
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

struct ndt_type_callable_wrapper {
    dynd::ndt::type d;
    dynd::gfunc::callable c;
    std::string funcname;
};

/**
 * This is the typeobject and struct of w_ndt_type_callable from Cython.
 */
extern PyTypeObject *WTypeCallable_Type;
inline bool WTypeCallable_CheckExact(PyObject *obj) {
    return Py_TYPE(obj) == WTypeCallable_Type;
}
inline bool WTypeCallable_Check(PyObject *obj) {
    return PyObject_TypeCheck(obj, WTypeCallable_Type);
}
struct WTypeCallable {
  PyObject_HEAD;
  // This is ndt_type_placement_wrapper in Cython-land
  ndt_type_callable_wrapper v;
};
void init_w_ndt_type_callable_typeobject(PyObject *type);
/**
 * This calls the callable in the ndt_type_callable_wrapper, which was
 * returned as a property of a dynd type.
 */
PyObject *ndt_type_callable_call(const ndt_type_callable_wrapper& ccw, PyObject *args, PyObject *kwargs);

/**
 * This calls the dynamic __construct__ function attached to the dynd type.
 */
PyObject *call_ndt_type_constructor_function(const dynd::ndt::type& dt, PyObject *args, PyObject *kwargs);

/**
 * Adds all the dynamic names exposed by the dynd type to the provided dict.
 */
void add_ndt_type_names_to_dir_dict(const dynd::ndt::type& dt, PyObject *dict);
/**
 * Retrieves a dynamic property from the dynd type as a Python object.
 */
PyObject *get_ndt_type_dynamic_property(const dynd::ndt::type& dt, PyObject *name);

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
 * Calls the callable with the single dynd type parameter
 *
 * \param funcname  The callable name.
 * \param c  The callable.
 * \param dt  The first parameter for the callable.
 */
PyObject *call_gfunc_callable(const std::string& funcname, const dynd::gfunc::callable& c, const dynd::ndt::type& dt);

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
 * Returns a wrapper for the callable with the dynd type as the first parameter.
 *
 * \param funcname  The callable name.
 * \param c  The callable.
 * \param d  The first parameter for the callable.
 */
PyObject *wrap_ndt_type_callable(const std::string& funcname, const dynd::gfunc::callable& c, const dynd::ndt::type& d);

} // namespace pydynd

#endif // _DYND__GFUNC_CALLABLE_FUNCTIONS_HPP_


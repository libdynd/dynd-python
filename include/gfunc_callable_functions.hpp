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

struct ndobject_callable_wrapper {
    dynd::ndobject n;
    dynd::gfunc::callable c;
    std::string funcname;
};

/**
 * This is the typeobject and struct of w_ndobject_callable from Cython.
 */
extern PyTypeObject *WNDObjectCallable_Type;
inline bool WNDObjectCallable_CheckExact(PyObject *obj) {
    return Py_TYPE(obj) == WNDObjectCallable_Type;
}
inline bool WNDObjectCallable_Check(PyObject *obj) {
    return PyObject_TypeCheck(obj, WNDObjectCallable_Type);
}
struct WNDObjectCallable {
  PyObject_HEAD;
  // This is ndobject_placement_wrapper in Cython-land
  ndobject_callable_wrapper v;
};
void init_w_ndobject_callable_typeobject(PyObject *type);

/**
 * This calls the callable in the ndobject_callable_wrapper, which was
 * returned as a property of an ndobject.
 */
PyObject *ndobject_callable_call(const ndobject_callable_wrapper& ncw, PyObject *args, PyObject *kwargs);

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
 * Adds all the dynamic names exposed by the ndobject to the provided dict.
 */
void add_ndobject_names_to_dir_dict(const dynd::ndobject& n, PyObject *dict);
/**
 * Retrieves a dynamic property from the ndobject.
 */
PyObject *get_ndobject_dynamic_property(const dynd::ndobject& n, PyObject *name);
/**
 * Sets a dynamic property of the ndobject.
 */
void set_ndobject_dynamic_property(const dynd::ndobject& n, PyObject *name, PyObject *value);

/**
 * Calls the callable with the single dtype parameter
 *
 * \param funcname  The callable name.
 * \param c  The callable.
 * \param dt  The first parameter for the callable.
 */
PyObject *call_gfunc_callable(const std::string& funcname, const dynd::gfunc::callable& c, const dynd::dtype& dt);

/**
 * Calls the callable with the single ndobject parameter
 *
 * \param funcname  The callable name.
 * \param c  The callable.
 * \param n  The first parameter for the callable.
 */
dynd::ndobject call_gfunc_callable(const std::string& funcname, const dynd::gfunc::callable& c, const dynd::ndobject& n);

/**
 * Returns a wrapper for the callable with the ndobject as the first parameter.
 *
 * \param funcname  The callable name.
 * \param c  The callable.
 * \param n  The first parameter for the callable.
 */
PyObject *wrap_ndobject_callable(const std::string& funcname, const dynd::gfunc::callable& c, const dynd::ndobject& n);

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


//
// Copyright (C) 2011-12, Dynamic NDArray Developers
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
PyObject *call_gfunc_callable(const std::string& funcname, const dynd::gfunc::callable& c, const dynd::ndobject& n);

/**
 * Returns a wrapper for the callable with the ndobject as the first parameter.
 *
 * \param funcname  The callable name.
 * \param c  The callable.
 * \param n  The first parameter for the callable.
 */
PyObject *wrap_ndobject_callable(const std::string& funcname, const dynd::gfunc::callable& c, const dynd::ndobject& n);

} // namespace pydynd

#endif // _DYND__GFUNC_CALLABLE_FUNCTIONS_HPP_
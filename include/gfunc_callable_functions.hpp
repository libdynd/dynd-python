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

void add_dtype_names_to_dir_dict(const dynd::dtype& dt, PyObject *dict);
PyObject *get_dtype_dynamic_property(const dynd::dtype& dt, PyObject *name);

void add_ndobject_names_to_dir_dict(const dynd::ndobject& n, PyObject *dict);
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
 * \param dt  The first parameter for the callable.
 */
PyObject *call_gfunc_callable(const std::string& funcname, const dynd::gfunc::callable& c, const dynd::ndobject& n);

} // namespace pydynd

#endif // _DYND__GFUNC_CALLABLE_FUNCTIONS_HPP_
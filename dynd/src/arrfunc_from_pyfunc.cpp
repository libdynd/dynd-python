//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>

#include <vector>

#include <dynd/array.hpp>
#include <dynd/kernels/base_kernel.hpp>
#include <dynd/func/callable.hpp>

#include <array_functions.hpp>
#include <utility_functions.hpp>
#include <arrfunc_from_pyfunc.hpp>
#include <type_functions.hpp>
#include <exception_translation.hpp>
#include <array_assign_from_py.hpp>
#include <kernels/apply_pyobject_kernel.hpp>

using namespace std;

dynd::nd::callable pydynd::nd::functional::apply(PyObject *instantiate_pyfunc,
                                                 const dynd::ndt::type &proto)
{
  if (proto.get_type_id() != dynd::callable_type_id) {
    stringstream ss;
    ss << "creating a dynd dynd::nd::callable from a python func requires a "
          "function "
          "prototype, was given type " << proto;
    throw dynd::type_error(ss.str());
  }

  Py_INCREF(instantiate_pyfunc);
  return dynd::nd::callable::make<apply_pyobject_kernel>(proto,
                                                         instantiate_pyfunc, 0);
}

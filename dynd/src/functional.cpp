//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "functional.hpp"
#include "kernels/apply_pyobject_kernel.hpp"
#include "callable_api.h"

using namespace std;
using namespace dynd;

nd::callable apply(const ndt::type &tp, PyObject *func)
{
  return nd::callable::make<apply_pyobject_kernel>(
      tp, apply_pyobject_kernel::static_data_type(func));
}

dynd::nd::callable &dynd_nd_callable_to_cpp_ref(PyObject *o) {
  if(dynd_nd_callable_to_ptr == NULL) {
    import_dynd__nd__callable();
  }
  return *dynd_nd_callable_to_ptr(reinterpret_cast<dynd_nd_callable_pywrapper*>(o));
}

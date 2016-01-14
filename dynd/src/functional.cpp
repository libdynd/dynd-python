//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "functional.hpp"
#include "kernels/apply_pyobject_kernel.hpp"

using namespace std;
using namespace dynd;

nd::callable apply(const ndt::type &tp, PyObject *func)
{
  return nd::callable::make<apply_pyobject_kernel>(
      tp, apply_pyobject_kernel::static_data_type(func));
}

//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include "type_functions.hpp"

#include <dynd/func/arrfunc.hpp>

namespace pydynd {
namespace nd {
  namespace functional {

    dynd::nd::arrfunc apply(PyObject *pyfunc, const dynd::ndt::type &proto);

    inline dynd::nd::arrfunc apply(PyObject *pyfunc, PyObject *proto)
    {
      return apply(pyfunc, make__type_from_pyobject(proto));
    }

  } // namespace pydynd::nd::functional
} // namespace pydynd::nd
} // namespace pydynd
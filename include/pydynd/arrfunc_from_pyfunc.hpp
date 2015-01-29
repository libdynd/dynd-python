//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__ARRFUNC_FROM_PYFUNC_HPP_
#define _DYND__ARRFUNC_FROM_PYFUNC_HPP_

#include "type_functions.hpp"

#include <dynd/func/arrfunc.hpp>

namespace pydynd {

dynd::nd::arrfunc arrfunc_from_pyfunc(PyObject *pyfunc,
                                      const dynd::ndt::type &proto);

inline dynd::nd::arrfunc arrfunc_from_pyfunc(PyObject *pyfunc, PyObject *proto)
{
  return arrfunc_from_pyfunc(pyfunc, make_ndt_type_from_pyobject(proto));
}

} // namespace pydynd

#endif // _DYND__ARRFUNC_FROM_PYFUNC_HPP_

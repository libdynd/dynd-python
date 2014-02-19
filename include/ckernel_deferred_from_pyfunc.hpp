//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
#ifndef _DYND__CKERNEL_DEFERRED_FROM_PYFUNC_HPP_
#define _DYND__CKERNEL_DEFERRED_FROM_PYFUNC_HPP_

namespace pydynd {

PyObject *ckernel_deferred_from_pyfunc(PyObject *instantiate_pyfunc, PyObject *types);

} // namespace pydynd

#endif // _DYND__CKERNEL_DEFERRED_FROM_PYFUNC_HPP_
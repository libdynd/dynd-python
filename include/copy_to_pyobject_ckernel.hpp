//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef PYDYND_COPY_TO_PYOBJECT_CKERNEL_HPP
#define PYDYND_COPY_TO_PYOBJECT_CKERNEL_HPP

#include <Python.h>

#include <dynd/kernels/ckernel_builder.hpp>

namespace pydynd {

intptr_t make_copy_to_pyobject_kernel(dynd::ckernel_builder *ckb,
                                      intptr_t ckb_offset,
                                      const dynd::ndt::type &src_tp,
                                      const char *src_arrmeta,
                                      bool struct_as_pytuple,
                                      dynd::kernel_request_t kernreq,
                                      const dynd::eval::eval_context *ectx);

} // namespace pydynd

#endif // PYDYND_COPY_TO_PYOBJECT_CKERNEL_HPP

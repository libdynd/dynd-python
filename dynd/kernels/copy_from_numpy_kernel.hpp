#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/base_virtual_kernel.hpp>

#include "config.hpp"

namespace pydynd {
namespace nd {

  struct copy_from_numpy_kernel : base_virtual_kernel<copy_from_numpy_kernel> {
    static intptr_t
    instantiate(const arrfunc_type_data *af, const ndt::arrfunc_type *af_tp,
                char *data, void *ckb, intptr_t ckb_offset,
                const ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                const ndt::type *src_tp, const char *const *src_arrmeta,
                kernel_request_t kernreq, const eval::eval_context *ectx,
                const array &kwds,
                const std::map<nd::string, ndt::type> &tp_vars);
  };

} // namespace pydynd::nd
} // namespace pydynd
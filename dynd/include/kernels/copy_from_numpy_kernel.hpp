#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/kernels/base_virtual_kernel.hpp>

#include "config.hpp"

namespace pydynd {
namespace nd {

  struct copy_from_numpy_kernel
      : dynd::nd::base_virtual_kernel<copy_from_numpy_kernel> {
    static intptr_t
    instantiate(char *static_data, size_t data_size, char *data, void *ckb,
                intptr_t ckb_offset, const dynd::ndt::type &dst_tp,
                const char *dst_arrmeta, intptr_t nsrc,
                const dynd::ndt::type *src_tp, const char *const *src_arrmeta,
                dynd::kernel_request_t kernreq,
                const dynd::eval::eval_context *ectx, intptr_t nkwd,
                const dynd::nd::array *kwds,
                const std::map<std::string, dynd::ndt::type> &tp_vars);
  };

} // namespace pydynd::nd
} // namespace pydynd

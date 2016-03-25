#pragma once

#include <dynd/callables/base_callable.hpp>

namespace pydynd {
namespace nd {

  class copy_from_numpy_callable : public dynd::nd::base_callable {
  public:
    copy_from_numpy_callable() : dynd::nd::base_callable(dynd::ndt::type("(void, broadcast: bool) -> T")) {}

    void instantiate(char *static_data, char *data, dynd::nd::kernel_builder *ckb, const dynd::ndt::type &dst_tp,
                     const char *dst_arrmeta, intptr_t nsrc, const dynd::ndt::type *src_tp,
                     const char *const *src_arrmeta, dynd::kernel_request_t kernreq, intptr_t nkwd,
                     const dynd::nd::array *kwds, const std::map<std::string, dynd::ndt::type> &tp_vars);
  };

} // namespace pydynd::nd
} // namespace pydynd

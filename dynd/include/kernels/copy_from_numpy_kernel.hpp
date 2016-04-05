#pragma once

#include <dynd/callables/base_callable.hpp>

namespace pydynd {
namespace nd {

  class copy_from_numpy_callable : public dynd::nd::base_callable {
  public:
    copy_from_numpy_callable() : dynd::nd::base_callable(dynd::ndt::type("(void, broadcast: bool) -> T")) {}

    dynd::ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data),
                            dynd::nd::call_graph &DYND_UNUSED(cg), const dynd::ndt::type &dst_tp,
                            size_t DYND_UNUSED(nsrc), const dynd::ndt::type *DYND_UNUSED(src_tp),
                            size_t DYND_UNUSED(nkwd), const dynd::nd::array *DYND_UNUSED(kwds),
                            const std::map<std::string, dynd::ndt::type> &DYND_UNUSED(tp_vars))
    {
      return dst_tp;
    }

    void instantiate(dynd::nd::call_node *&DYND_UNUSED(node), char *data, dynd::nd::kernel_builder *ckb,
                     const dynd::ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                     const dynd::ndt::type *src_tp, const char *const *src_arrmeta, dynd::kernel_request_t kernreq,
                     intptr_t nkwd, const dynd::nd::array *kwds, const std::map<std::string, dynd::ndt::type> &tp_vars);
  };

} // namespace pydynd::nd
} // namespace pydynd

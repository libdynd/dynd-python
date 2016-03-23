#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <array_functions.hpp>
#include <utility_functions.hpp>

#include "type_functions.hpp"
#include "array_conversions.hpp"

namespace pydynd {
namespace nd {
  namespace functional {

    struct apply_jit_kernel : dynd::nd::base_strided_kernel<apply_jit_kernel> {
      typedef void (*func_type)(char *dst, char *const *src);

      intptr_t nsrc;
      func_type func;

      apply_jit_kernel(intptr_t nsrc, func_type func) : nsrc(nsrc), func(func)
      {
      }

      void call(dynd::nd::array *dst, const dynd::nd::array *src)
      {
        std::vector<char *> src_data(nsrc);
        for (int i = 0; i < nsrc; ++i) {
          src_data[i] = const_cast<char *>(src[i].cdata());
        }

        single(const_cast<char *>(dst->cdata()), src_data.data());
      }

      void single(char *dst, char *const *src) { func(dst, src); }

      void strided(char *dst, intptr_t dst_stride, char *const *src,
                   const intptr_t *src_stride, size_t count)
      {
        std::vector<char *> src_copy(nsrc);
        memcpy(src_copy.data(), src, nsrc * sizeof(char *));
        for (size_t i = 0; i != count; ++i) {
          single(dst, src_copy.data());
          dst += dst_stride;
          for (int j = 0; j < nsrc; ++j) {
            src_copy[j] += src_stride[j];
          }
        }
      }

      static void
      instantiate(char *static_data, char *DYND_UNUSED(data),
                  dynd::nd::kernel_builder *ckb, const dynd::ndt::type &dst_tp,
                  const char *dst_arrmeta, intptr_t nsrc,
                  const dynd::ndt::type *src_tp, const char *const *src_arrmeta,
                  dynd::kernel_request_t kernreq, intptr_t nkwd,
                  const dynd::nd::array *kwds,
                  const std::map<std::string, dynd::ndt::type> &tp_vars)
      {
        ckb->emplace_back<apply_jit_kernel>(
            kernreq, nsrc, *reinterpret_cast<func_type *>(static_data));
      }
    };

  } // namespace pydynd::nd::functional
} // namespace pydynd::nd
} // namespace pydynd

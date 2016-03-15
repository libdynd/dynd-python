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

    inline dynd::nd::callable apply_jit(const dynd::ndt::type &tp,
                                        intptr_t func)
    {
      return dynd::nd::callable::make<apply_jit_kernel>(
          tp, reinterpret_cast<apply_jit_kernel::func_type *>(func));
    }

    struct jit_dispatcher {
      typedef PyObject *(*jit_type)(PyObject *func, intptr_t nsrc,
                                    const dynd::ndt::type *src_tp);

      PyObject *func;
      jit_type jit;
      std::map<std::vector<dynd::type_id_t>, PyObject *> children;

      jit_dispatcher(PyObject *func, jit_type jit)
          : func((Py_INCREF(func), func)), jit(jit)
      {
      }

      ~jit_dispatcher()
      {
        // Need to handle dangling references here
        // Py_DECREF(func);
      }

      dynd::nd::callable &operator()(const dynd::ndt::type &DYND_UNUSED(dst_tp),
                                     intptr_t nsrc,
                                     const dynd::ndt::type *src_tp)
      {
        std::vector<dynd::type_id_t> key(nsrc);
        for (int i = 0; i < nsrc; ++i) {
          key[i] = src_tp[i].get_id();
        }

        PyObject *&obj = children[key];
        if (obj == NULL) {

          obj = (*jit)(func, nsrc, src_tp);
          if (PyErr_Occurred()) {
            throw std::runtime_error("An exception was raised in Python");
          }

          Py_INCREF(obj);
        }
        return callable_to_cpp_ref(obj);
      }
    };

  } // namespace pydynd::nd::functional
} // namespace pydynd::nd
} // namespace pydynd

#pragma once

#include "config.hpp"
#include <dynd/kernels/base_kernel.hpp>
#include <array_functions.hpp>
#include <utility_functions.hpp>
#include <arrfunc_from_pyfunc.hpp>
#include <type_functions.hpp>
#include <exception_translation.hpp>
#include <array_assign_from_py.hpp>

#include "../arrfunc_functions.hpp"

namespace pydynd {
namespace nd {
  namespace functional {

    struct apply_jit_kernel : dynd::nd::base_kernel<apply_jit_kernel> {
      typedef void (*func_type)(char *dst, char *const *src);

      intptr_t nsrc;
      func_type func;

      apply_jit_kernel(intptr_t nsrc, func_type func) : nsrc(nsrc), func(func)
      {
      }

      void single(char *dst, char *const *src)
      {
        func(dst, src);
      }

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

      static intptr_t instantiate(
          char *static_data, size_t DYND_UNUSED(data_size),
          char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
          const dynd::ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
          const dynd::ndt::type *src_tp, const char *const *src_arrmeta,
          dynd::kernel_request_t kernreq, const dynd::eval::eval_context *ectx,
          intptr_t nkwd, const dynd::nd::array *kwds,
          const std::map<std::string, dynd::ndt::type> &tp_vars)
      {
        apply_jit_kernel::make(ckb, kernreq, ckb_offset, nsrc,
                               *reinterpret_cast<func_type *>(static_data));
        return ckb_offset;
      }
    };

    inline dynd::nd::callable apply_jit(const dynd::ndt::type &tp,
                                        intptr_t func)
    {
      return dynd::nd::callable::make<apply_jit_kernel>(
          tp, reinterpret_cast<apply_jit_kernel::func_type *>(func), 0);
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
          key[i] = src_tp[i].get_type_id();
        }

        PyObject *&obj = children[key];
        if (obj == NULL) {

          obj = (*jit)(func, nsrc, src_tp);
          if (obj == NULL) {
            PyErr_Print();
            throw std::runtime_error("An exception was raised in Python");
          }

          Py_INCREF(obj);
        }

        return reinterpret_cast<DyND_PyCallableObject *>(obj)->v;
      }
    };

  } // namespace pydynd::nd::functional
} // namespace pydynd::nd
} // namespace pydynd

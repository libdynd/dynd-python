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

    struct apply_numba_kernel
        : dynd::nd::base_kernel<apply_numba_kernel, dynd::kernel_request_host,
                                2> {
      typedef apply_numba_kernel self_type;

      void (*func)(char *dst, char *const *src);

      apply_numba_kernel(void (*func)(char *dst, char *const *src)) : func(func)
      {
      }

      void single(char *dst, char *const *src)
      {
        std::cout << "in single" << std::endl;
        func(dst, src);
      }

      static intptr_t instantiate(
          char *static_data, size_t DYND_UNUSED(data_size),
          char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
          const dynd::ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
          const dynd::ndt::type *src_tp, const char *const *src_arrmeta,
          dynd::kernel_request_t kernreq, const dynd::eval::eval_context *ectx,
          const dynd::nd::array &kwds,
          const std::map<std::string, dynd::ndt::type> &tp_vars)
      {
        self_type *self = self_type::make(
            ckb, kernreq, ckb_offset,
            *reinterpret_cast<void (**)(char *dst, char *const *src)>(
                static_data));
        return ckb_offset;
      }
    };

    inline dynd::nd::callable numba_helper(const dynd::ndt::type &tp,
                                           intptr_t ptr)
    {
      return dynd::nd::callable::make<apply_numba_kernel>(
          tp, reinterpret_cast<void (*)(char *dst, char *const *src)>(ptr), 0);
    }

    template <typename T>
    inline T &dereference(T *ptr)
    {
      return *ptr;
    }

    struct jit_dispatcher {
      typedef PyObject *(*jit_type)(PyObject *func,
                                    const dynd::ndt::type &dst_tp,
                                    intptr_t nsrc,
                                    const dynd::ndt::type *src_tp);

      PyObject *func;
      jit_type jit;

      jit_dispatcher(PyObject *func, jit_type jit)
          : func((Py_INCREF(func), func)), jit(jit)
      {
      }

      ~jit_dispatcher() { Py_DECREF(func); }

      dynd::nd::callable &operator()(const dynd::ndt::type &dst_tp,
                                     intptr_t nsrc,
                                     const dynd::ndt::type *src_tp)
      {
        PyObject *obj = (*jit)(func, dst_tp, nsrc, src_tp);
        Py_INCREF(obj);

        return reinterpret_cast<DyND_PyCallableObject *>(obj)->v;
      }
    };

  } // namespace pydynd::nd::functional
} // namespace pydynd::nd
} // namespace pydynd
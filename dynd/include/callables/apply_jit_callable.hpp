#pragma once

#include "kernels/apply_jit_kernel.hpp"

#include <dynd/callables/base_dispatch_callable.hpp>

namespace pydynd {
namespace nd {
  namespace functional {

    class apply_jit_dispatch_callable : public dynd::nd::base_dispatch_callable {
    public:
      typedef PyObject *(*jit_type)(PyObject *func, intptr_t nsrc, const dynd::ndt::type *src_tp);

      PyObject *func;
      jit_type jit;
      std::map<std::vector<dynd::type_id_t>, PyObject *> children;

      apply_jit_dispatch_callable(const dynd::ndt::type &tp, PyObject *func, jit_type jit)
          : dynd::nd::base_dispatch_callable(tp), func((Py_INCREF(func), func)), jit(jit)
      {
      }

      dynd::ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), dynd::nd::call_graph &cg,
                              const dynd::ndt::type &dst_tp, size_t DYND_UNUSED(nsrc),
                              const dynd::ndt::type *DYND_UNUSED(src_tp), size_t DYND_UNUSED(nkwd),
                              const dynd::nd::array *DYND_UNUSED(kwds),
                              const std::map<std::string, dynd::ndt::type> &DYND_UNUSED(tp_vars))
      {
        return dst_tp;
      }

      ~apply_jit_dispatch_callable()
      {
        // Need to handle dangling references here
        // Py_DECREF(func);
      }

      const dynd::nd::callable &specialize(const dynd::ndt::type &DYND_UNUSED(dst_tp), intptr_t nsrc,
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

    inline dynd::nd::callable apply_jit(const dynd::ndt::type &DYND_UNUSED(tp), intptr_t DYND_UNUSED(func))
    {
      throw std::runtime_error("apply_jit is temporarily disabled");
      //      return dynd::nd::make_callable<apply_jit_dispatch_callable>(
      //        tp, nullptr, reinterpret_cast<apply_jit_kernel::func_type *>(func));
    }

  } // namespace pydynd::nd::functional
} // namespace pydynd::nd
} // namespace pydynd

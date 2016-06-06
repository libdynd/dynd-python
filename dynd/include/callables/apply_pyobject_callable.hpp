#pragma once

#include "kernels/apply_pyobject_kernel.hpp"

namespace pydynd {
namespace nd {
  namespace functional {

    class apply_pyobject_callable : public dynd::nd::base_callable {
    public:
      PyObject *func;

      apply_pyobject_callable(const dynd::ndt::type &tp, PyObject *func) : dynd::nd::base_callable(tp), func(func)
      {
        Py_INCREF(func);
      }

      dynd::ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data),
                              dynd::nd::call_graph &cg, const dynd::ndt::type &dst_tp, size_t DYND_UNUSED(nsrc),
                              const dynd::ndt::type *DYND_UNUSED(src_tp), size_t DYND_UNUSED(nkwd),
                              const dynd::nd::array *DYND_UNUSED(kwds),
                              const std::map<std::string, dynd::ndt::type> &DYND_UNUSED(tp_vars))
      {
        return dst_tp;
      }

      /*
       void instantiate(dynd::nd::call_node *&node, char *DYND_UNUSED(data), dynd::nd::kernel_builder *ckb,
                        const dynd::ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                        const dynd::ndt::type *src_tp, const char *const *src_arrmeta, dynd::kernel_request_t kernreq,
                        intptr_t nkwd, const dynd::nd::array *kwds,
                        const std::map<std::string, dynd::ndt::type> &tp_vars)
       {
         pydynd::with_gil pgs;

         std::vector<dynd::ndt::type> src_tp_copy(nsrc);
         for (int i = 0; i < nsrc; ++i) {
           src_tp_copy[i] = src_tp[i];
         }

         intptr_t ckb_offset = ckb->size();
         ckb->emplace_back<apply_pyobject_kernel>(kernreq);
         apply_pyobject_kernel *self = ckb->get_at<apply_pyobject_kernel>(ckb_offset);
         self->m_proto = dynd::ndt::callable_type::make(dst_tp, src_tp_copy);
         self->m_pyfunc = func;
         Py_XINCREF(self->m_pyfunc);
         self->m_dst_arrmeta = dst_arrmeta;
         self->m_src_arrmeta.resize(nsrc);
         copy(src_arrmeta, src_arrmeta + nsrc, self->m_src_arrmeta.begin());

         dynd::ndt::type child_src_tp = dynd::ndt::make_type<pyobject_type>();
         dynd::nd::assign->instantiate(node, nullptr, ckb, dst_tp, dst_arrmeta, 1, &child_src_tp, nullptr,
                                       dynd::kernel_request_single, 0, nullptr, tp_vars);
       }
       */
    };

  } // namespace pydynd::nd::functional
} // namespace pydynd::nd
} // namespace pydynd

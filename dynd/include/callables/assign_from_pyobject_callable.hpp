#pragma once

#include "kernels/assign_from_pyobject_kernel.hpp"
#include <dynd/callables/base_callable.hpp>

using namespace dynd;

namespace pydynd {
namespace nd {

  template <typename ReturnType>
  class assign_from_pyobject_callable
      : public dynd::nd::default_instantiable_callable<assign_from_pyobject_kernel<ReturnType>> {
  public:
    assign_from_pyobject_callable()
        : dynd::nd::default_instantiable_callable<assign_from_pyobject_kernel<ReturnType>>(
              dynd::ndt::make_type<dynd::ndt::callable_type>(ndt::make_type<ReturnType>(),
                                                             {dynd::ndt::make_type<pyobject_type>()}))
    {
    }
  };

  template <>
  class assign_from_pyobject_callable<bytes> : public dynd::nd::base_callable {
  public:
    assign_from_pyobject_callable()
        : dynd::nd::base_callable(dynd::ndt::make_type<dynd::ndt::callable_type>(
              ndt::make_type<bytes>(), {dynd::ndt::make_type<pyobject_type>()}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), dynd::nd::call_graph &cg,
                      const dynd::ndt::type &dst_tp, size_t DYND_UNUSED(nsrc),
                      const dynd::ndt::type *DYND_UNUSED(src_tp), size_t DYND_UNUSED(nkwd),
                      const dynd::nd::array *DYND_UNUSED(kwds),
                      const std::map<std::string, dynd::ndt::type> &DYND_UNUSED(tp_vars))
    {
      return dst_tp;
    }

    /*
        void instantiate(dynd::nd::call_node *&DYND_UNUSED(node), char *data, dynd::nd::kernel_builder *ckb,
                         const dynd::ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                         const dynd::ndt::type *src_tp, const char *const *src_arrmeta, dynd::kernel_request_t kernreq,
                         intptr_t nkwd, const dynd::nd::array *kwds, const std::map<std::string, dynd::ndt::type>
       &tp_vars)
        {
          kb.emplace_back<assign_from_pyobject_kernel<bytes_id>>(kernreq, dst_tp, dst_arrmeta);
        }
    */
  };

  template <>
  class assign_from_pyobject_callable<ndt::fixed_bytes_type> : public dynd::nd::base_callable {
  public:
    assign_from_pyobject_callable()
        : dynd::nd::base_callable(dynd::ndt::make_type<dynd::ndt::callable_type>(
              ndt::make_type<ndt::fixed_bytes_kind_type>(), {dynd::ndt::make_type<pyobject_type>()}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), dynd::nd::call_graph &cg,
                      const dynd::ndt::type &dst_tp, size_t DYND_UNUSED(nsrc),
                      const dynd::ndt::type *DYND_UNUSED(src_tp), size_t DYND_UNUSED(nkwd),
                      const dynd::nd::array *DYND_UNUSED(kwds),
                      const std::map<std::string, dynd::ndt::type> &DYND_UNUSED(tp_vars))
    {
      cg.emplace_back([dst_tp](dynd::nd::kernel_builder &kb, dynd::kernel_request_t kernreq, char *DYND_UNUSED(data),
                               const char *dst_arrmeta, size_t nsrc, const char *const *src_arrmeta) {
        kb.emplace_back<assign_from_pyobject_kernel<ndt::fixed_bytes_type>>(kernreq, dst_tp, dst_arrmeta);
      });

      return dst_tp;
    }
  };

  template <>
  class assign_from_pyobject_callable<string> : public dynd::nd::base_callable {
  public:
    assign_from_pyobject_callable()
        : dynd::nd::base_callable(dynd::ndt::make_type<dynd::ndt::callable_type>(
              ndt::make_type<string>(), {dynd::ndt::make_type<pyobject_type>()}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), dynd::nd::call_graph &cg,
                      const dynd::ndt::type &dst_tp, size_t DYND_UNUSED(nsrc),
                      const dynd::ndt::type *DYND_UNUSED(src_tp), size_t DYND_UNUSED(nkwd),
                      const dynd::nd::array *DYND_UNUSED(kwds),
                      const std::map<std::string, dynd::ndt::type> &DYND_UNUSED(tp_vars))
    {
      cg.emplace_back([dst_tp](dynd::nd::kernel_builder &kb, dynd::kernel_request_t kernreq, char *DYND_UNUSED(data),
                               const char *dst_arrmeta, size_t nsrc, const char *const *src_arrmeta) {
        kb.emplace_back<assign_from_pyobject_kernel<dynd::string>>(kernreq, dst_tp, dst_arrmeta);
      });

      return dst_tp;
    }
  };

  template <>
  class assign_from_pyobject_callable<ndt::fixed_string_type> : public dynd::nd::base_callable {
  public:
    assign_from_pyobject_callable()
        : dynd::nd::base_callable(dynd::ndt::make_type<dynd::ndt::callable_type>(
              ndt::make_type<ndt::fixed_string_kind_type>(), {dynd::ndt::make_type<pyobject_type>()}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), dynd::nd::call_graph &cg,
                      const dynd::ndt::type &dst_tp, size_t DYND_UNUSED(nsrc),
                      const dynd::ndt::type *DYND_UNUSED(src_tp), size_t DYND_UNUSED(nkwd),
                      const dynd::nd::array *DYND_UNUSED(kwds),
                      const std::map<std::string, dynd::ndt::type> &DYND_UNUSED(tp_vars))
    {
      cg.emplace_back([dst_tp](dynd::nd::kernel_builder &kb, dynd::kernel_request_t kernreq, char *DYND_UNUSED(data),
                               const char *dst_arrmeta, size_t DYND_UNUSED(nsrc),
                               const char *const *DYND_UNUSED(src_arrmeta)) {
        kb.emplace_back<assign_from_pyobject_kernel<dynd::ndt::fixed_string_type>>(kernreq, dst_tp, dst_arrmeta);
      });

      return dst_tp;
    }
  };

  template <>
  class assign_from_pyobject_callable<ndt::option_type> : public dynd::nd::base_callable {
  public:
    assign_from_pyobject_callable()
        : dynd::nd::base_callable(dynd::ndt::make_type<dynd::ndt::callable_type>(
              ndt::make_type<ndt::option_type>(ndt::make_type<ndt::any_kind_type>()),
              {dynd::ndt::make_type<pyobject_type>()}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), dynd::nd::call_graph &cg,
                      const dynd::ndt::type &dst_tp, size_t nsrc, const dynd::ndt::type *src_tp, size_t nkwd,
                      const dynd::nd::array *kwds, const std::map<std::string, dynd::ndt::type> &tp_vars)
    {
      cg.emplace_back([dst_tp](dynd::nd::kernel_builder &kb, dynd::kernel_request_t kernreq, char *DYND_UNUSED(data),
                               const char *dst_arrmeta, size_t nsrc, const char *const *src_arrmeta) {
        intptr_t root_ckb_offset = kb.size();
        kb.emplace_back<assign_from_pyobject_kernel<dynd::ndt::option_type>>(kernreq, dst_tp, dst_arrmeta);
        intptr_t ckb_offset = kb.size();
        kb(dynd::kernel_request_single, nullptr, dst_arrmeta, nsrc, nullptr);

        ckb_offset = kb.size();
        kb.get_at<assign_from_pyobject_kernel<ndt::option_type>>(root_ckb_offset)->copy_value_offset =
            ckb_offset - root_ckb_offset;
        kb(dynd::kernel_request_single, nullptr, dst_arrmeta, nsrc, src_arrmeta);

        ckb_offset = kb.size();
      });

      dynd::nd::assign_na->resolve(this, nullptr, cg, dst_tp, nsrc, nullptr, nkwd, kwds, tp_vars);
      dynd::nd::assign->resolve(this, nullptr, cg, dst_tp.extended<dynd::ndt::option_type>()->get_value_type(), nsrc,
                                src_tp, nkwd, kwds, tp_vars);

      return dst_tp;
    }

    /*
        void instantiate(dynd::nd::call_node *&node, char *data, dynd::nd::kernel_builder *ckb,
                         const dynd::ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                         const dynd::ndt::type *src_tp, const char *const *src_arrmeta, dynd::kernel_request_t kernreq,
                         intptr_t nkwd, const dynd::nd::array *kwds, const std::map<std::string, dynd::ndt::type>
       &tp_vars)
        {
          intptr_t root_ckb_offset = kb.size();
          kb.emplace_back<assign_from_pyobject_kernel<option_id>>(kernreq, dst_tp, dst_arrmeta);
          intptr_t ckb_offset = kb.size();
          dynd::nd::assign_na->instantiate(node, NULL, ckb, dst_tp, dst_arrmeta, nsrc, NULL, NULL,
                                           dynd::kernel_request_single, nkwd, kwds, tp_vars);
          ckb_offset = kb.size();
          kb->get_at<assign_from_pyobject_kernel<option_id>>(root_ckb_offset)->copy_value_offset =
              ckb_offset - root_ckb_offset;
          dynd::nd::assign->instantiate(node, NULL, ckb, dst_tp.extended<dynd::ndt::option_type>()->get_value_type(),
                                        dst_arrmeta, nsrc, src_tp, src_arrmeta, dynd::kernel_request_single, nkwd, kwds,
                                        tp_vars);
          ckb_offset = kb.size();
        }
    */
  };

  template <>
  class assign_from_pyobject_callable<ndt::tuple_type> : public dynd::nd::base_callable {
  public:
    assign_from_pyobject_callable()
        : dynd::nd::base_callable(dynd::ndt::make_type<dynd::ndt::callable_type>(
              ndt::make_type<ndt::tuple_type>(true), {dynd::ndt::make_type<pyobject_type>()}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), dynd::nd::call_graph &cg,
                      const dynd::ndt::type &dst_tp, size_t nsrc, const dynd::ndt::type *src_tp, size_t nkwd,
                      const dynd::nd::array *kwds, const std::map<std::string, dynd::ndt::type> &tp_vars)
    {
      intptr_t field_count = dst_tp.extended<dynd::ndt::tuple_type>()->get_field_count();
      const std::vector<dynd::ndt::type> &field_types = dst_tp.extended<dynd::ndt::tuple_type>()->get_field_types();

      const std::vector<uintptr_t> &arrmeta_offsets = dst_tp.extended<dynd::ndt::tuple_type>()->get_arrmeta_offsets();

      cg.emplace_back([arrmeta_offsets, dst_tp, field_count](
          dynd::nd::kernel_builder &kb, dynd::kernel_request_t kernreq, char *DYND_UNUSED(data),
          const char *dst_arrmeta, size_t nsrc, const char *const *src_arrmeta) {
        intptr_t ckb_offset = kb.size();
        intptr_t root_ckb_offset = ckb_offset;
        kb.emplace_back<assign_from_pyobject_kernel<ndt::tuple_type>>(kernreq);
        assign_from_pyobject_kernel<ndt::tuple_type> *self =
            kb.get_at<assign_from_pyobject_kernel<ndt::tuple_type>>(root_ckb_offset);
        ckb_offset = kb.size();
        self->m_dst_tp = dst_tp;
        self->m_dst_arrmeta = dst_arrmeta;
        self->m_dim_broadcast = false;
        self->m_copy_el_offsets.resize(field_count);
        for (intptr_t i = 0; i < field_count; ++i) {
          kb.reserve(ckb_offset);
          self = kb.get_at<assign_from_pyobject_kernel<ndt::tuple_type>>(root_ckb_offset);
          self->m_copy_el_offsets[i] = ckb_offset - root_ckb_offset;
          const char *field_arrmeta = dst_arrmeta + arrmeta_offsets[i];
          kb(dynd::kernel_request_single, nullptr, field_arrmeta, nsrc, src_arrmeta);
          ckb_offset = kb.size();
        }
      });

      for (intptr_t i = 0; i < field_count; ++i) {
        dynd::nd::assign->resolve(this, nullptr, cg, field_types[i], nsrc, src_tp, nkwd, kwds, tp_vars);
      }

      return dst_tp;
    }

    /*
        void instantiate(dynd::nd::call_node *&node, char *data, dynd::nd::kernel_builder *ckb,
                         const dynd::ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                         const dynd::ndt::type *src_tp, const char *const *src_arrmeta, dynd::kernel_request_t kernreq,
                         intptr_t nkwd, const dynd::nd::array *kwds, const std::map<std::string, dynd::ndt::type>
       &tp_vars)
        {
          bool dim_broadcast = false;

          intptr_t ckb_offset = kb.size();
          intptr_t root_ckb_offset = ckb_offset;
          kb.emplace_back<assign_from_pyobject_kernel<tuple_id>>(kernreq);
          assign_from_pyobject_kernel<tuple_id> *self =
       kb.get_at<assign_from_pyobject_kernel<tuple_id>>(root_ckb_offset);
          ckb_offset = kb.size();
          self->m_dst_tp = dst_tp;
          self->m_dst_arrmeta = dst_arrmeta;
          intptr_t field_count = dst_tp.extended<dynd::ndt::tuple_type>()->get_field_count();
          const dynd::ndt::type *field_types = dst_tp.extended<dynd::ndt::tuple_type>()->get_field_types_raw();
          const uintptr_t *arrmeta_offsets = dst_tp.extended<dynd::ndt::tuple_type>()->get_arrmeta_offsets_raw();
          self->m_dim_broadcast = dim_broadcast;
          self->m_copy_el_offsets.resize(field_count);
          for (intptr_t i = 0; i < field_count; ++i) {
            kb.reserve(ckb_offset);
            self = kb.get_at<assign_from_pyobject_kernel<tuple_id>>(root_ckb_offset);
            self->m_copy_el_offsets[i] = ckb_offset - root_ckb_offset;
            const char *field_arrmeta = dst_arrmeta + arrmeta_offsets[i];
            dynd::nd::assign->instantiate(node, NULL, ckb, field_types[i], field_arrmeta, nsrc, src_tp, src_arrmeta,
                                          dynd::kernel_request_single, nkwd, kwds, tp_vars);
            ckb_offset = kb->size();
          }
        }
    */
  };

  template <>
  class assign_from_pyobject_callable<ndt::struct_type> : public dynd::nd::base_callable {
  public:
    assign_from_pyobject_callable()
        : dynd::nd::base_callable(dynd::ndt::make_type<dynd::ndt::callable_type>(
              ndt::make_type<ndt::struct_type>(true), {dynd::ndt::make_type<pyobject_type>()}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), dynd::nd::call_graph &cg,
                      const dynd::ndt::type &dst_tp, size_t nsrc, const dynd::ndt::type *src_tp, size_t nkwd,
                      const dynd::nd::array *kwds, const std::map<std::string, dynd::ndt::type> &tp_vars)
    {
      bool dim_broadcast = false;
      intptr_t field_count = dst_tp.extended<dynd::ndt::struct_type>()->get_field_count();
      const dynd::ndt::type *field_types = dst_tp.extended<dynd::ndt::struct_type>()->get_field_types_raw();
      const std::vector<uintptr_t> &arrmeta_offsets = dst_tp.extended<dynd::ndt::struct_type>()->get_arrmeta_offsets();

      cg.emplace_back([arrmeta_offsets, dim_broadcast, field_count,
                       dst_tp](dynd::nd::kernel_builder &kb, dynd::kernel_request_t kernreq, char *DYND_UNUSED(data),
                               const char *dst_arrmeta, size_t nsrc, const char *const *src_arrmeta) {
        intptr_t ckb_offset = kb.size();
        intptr_t root_ckb_offset = ckb_offset;

        kb.emplace_back<assign_from_pyobject_kernel<ndt::struct_type>>(kernreq);

        assign_from_pyobject_kernel<ndt::struct_type> *self =
            kb.get_at<assign_from_pyobject_kernel<ndt::struct_type>>(root_ckb_offset);
        ckb_offset = kb.size();

        self->m_dst_tp = dst_tp;
        self->m_dst_arrmeta = dst_arrmeta;
        self->m_dim_broadcast = dim_broadcast;
        self->m_copy_el_offsets.resize(field_count);

        for (intptr_t i = 0; i < field_count; ++i) {
          kb.reserve(ckb_offset);
          self = kb.get_at<assign_from_pyobject_kernel<ndt::struct_type>>(root_ckb_offset);
          self->m_copy_el_offsets[i] = ckb_offset - root_ckb_offset;
          const char *field_arrmeta = dst_arrmeta + arrmeta_offsets[i];
          kb(dynd::kernel_request_single, nullptr, field_arrmeta, nsrc, src_arrmeta);
          ckb_offset = kb.size();
        }
      });

      for (intptr_t i = 0; i < field_count; ++i) {
        dynd::nd::assign->resolve(this, nullptr, cg, field_types[i], nsrc, src_tp, nkwd, kwds, tp_vars);
      }

      return dst_tp;
    }

    /*
        void instantiate(dynd::nd::call_node *node, char *data, dynd::nd::kernel_builder *ckb,
                         const dynd::ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                         const dynd::ndt::type *src_tp, const char *const *src_arrmeta, dynd::kernel_request_t kernreq,
                         intptr_t nkwd, const dynd::nd::array *kwds, const std::map<std::string, dynd::ndt::type>
       &tp_vars)
        {
          bool dim_broadcast = false;

          intptr_t ckb_offset = kb.size();
          intptr_t root_ckb_offset = ckb_offset;
          kb.emplace_back<assign_from_pyobject_kernel<struct_id>>(kernreq);
          assign_from_pyobject_kernel<struct_id> *self =
       kb.get_at<assign_from_pyobject_kernel<struct_id>>(root_ckb_offset);
          ckb_offset = kb.size();
          self->m_dst_tp = dst_tp;
          self->m_dst_arrmeta = dst_arrmeta;
          intptr_t field_count = dst_tp.extended<dynd::ndt::struct_type>()->get_field_count();
          const dynd::ndt::type *field_types = dst_tp.extended<dynd::ndt::struct_type>()->get_field_types_raw();
          const uintptr_t *arrmeta_offsets = dst_tp.extended<dynd::ndt::struct_type>()->get_arrmeta_offsets_raw();
          self->m_dim_broadcast = dim_broadcast;
          self->m_copy_el_offsets.resize(field_count);
          for (intptr_t i = 0; i < field_count; ++i) {
            kb.reserve(ckb_offset);
            self = kb.get_at<assign_from_pyobject_kernel<struct_id>>(root_ckb_offset);
            self->m_copy_el_offsets[i] = ckb_offset - root_ckb_offset;
            const char *field_arrmeta = dst_arrmeta + arrmeta_offsets[i];
            dynd::nd::assign->instantiate(node, NULL, ckb, field_types[i], field_arrmeta, nsrc, src_tp, src_arrmeta,
                                          dynd::kernel_request_single, nkwd, kwds, tp_vars);
            ckb_offset = kb.size();
          }
        }
    */
  };

  template <>
  class assign_from_pyobject_callable<ndt::fixed_dim_type> : public dynd::nd::base_callable {
  public:
    assign_from_pyobject_callable()
        : dynd::nd::base_callable(dynd::ndt::make_type<dynd::ndt::callable_type>(
              ndt::type("Fixed * Any"), {dynd::ndt::make_type<pyobject_type>()}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), dynd::nd::call_graph &cg,
                      const dynd::ndt::type &dst_tp, size_t nsrc, const dynd::ndt::type *src_tp, size_t nkwd,
                      const dynd::nd::array *kwds, const std::map<std::string, dynd::ndt::type> &tp_vars)
    {
      dynd::ndt::type el_tp = dst_tp.extended<ndt::fixed_dim_type>()->get_element_type();

      cg.emplace_back([dst_tp](dynd::nd::kernel_builder &kb, dynd::kernel_request_t kernreq, char *DYND_UNUSED(data),
                               const char *dst_arrmeta, size_t nsrc, const char *const *src_arrmeta) {
        const char *el_arrmeta = dst_arrmeta + sizeof(size_stride_t);
        intptr_t dim_size = reinterpret_cast<const size_stride_t *>(dst_arrmeta)->dim_size;
        intptr_t stride = reinterpret_cast<const size_stride_t *>(dst_arrmeta)->stride;

        intptr_t ckb_offset = kb.size();
        intptr_t root_ckb_offset = ckb_offset;
        kb.emplace_back<assign_from_pyobject_kernel<ndt::fixed_dim_type>>(kernreq);

        assign_from_pyobject_kernel<ndt::fixed_dim_type> *self =
            kb.get_at<assign_from_pyobject_kernel<ndt::fixed_dim_type>>(root_ckb_offset);
        ckb_offset = kb.size();
        self->m_dim_size = dim_size;
        self->m_stride = stride;
        self->m_dst_tp = dst_tp;
        self->m_dst_arrmeta = dst_arrmeta;
        // from pyobject ckernel
        kb(dynd::kernel_request_strided, nullptr, el_arrmeta, nsrc, src_arrmeta);

        ckb_offset = kb.size();
        self = kb.get_at<assign_from_pyobject_kernel<ndt::fixed_dim_type>>(root_ckb_offset);
        self->m_copy_dst_offset = ckb_offset - root_ckb_offset;
        // dst to dst ckernel, for broadcasting case
        //        dynd::nd::array error_mode = assign_error_fractional;
        //      kb(dynd::kernel_request_strided, el_arrmeta, 1, &el_arrmeta);
        //    std::cout << "after second" << std::endl;
      });

      // from pyobject ckernel
      dynd::nd::assign->resolve(this, nullptr, cg, el_tp, nsrc, src_tp, nkwd, kwds, tp_vars);
      // dst to dst ckernel, for broadcasting case
      //      dynd::nd::array error_mode = assign_error_fractional;
      //    dynd::nd::assign->resolve(this, nullptr, cg, el_tp, 1, &el_tp, 1, &error_mode,
      //                            std::map<std::string, ndt::type>());

      return dst_tp;
    }
  };

  template <>
  class assign_from_pyobject_callable<ndt::var_dim_type> : public dynd::nd::base_callable {
  public:
    assign_from_pyobject_callable()
        : dynd::nd::base_callable(dynd::ndt::make_type<dynd::ndt::callable_type>(
              ndt::make_type<ndt::var_dim_type>(ndt::make_type<ndt::any_kind_type>()),
              {dynd::ndt::make_type<pyobject_type>()}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), dynd::nd::call_graph &cg,
                      const dynd::ndt::type &dst_tp, size_t nsrc, const dynd::ndt::type *src_tp, size_t nkwd,
                      const dynd::nd::array *kwds, const std::map<std::string, dynd::ndt::type> &tp_vars)
    {
      bool dim_broadcast = false;

      dynd::ndt::type el_tp = dst_tp.extended<dynd::ndt::var_dim_type>()->get_element_type();

      cg.emplace_back([dst_tp, el_tp](dynd::nd::kernel_builder &kb, dynd::kernel_request_t kernreq,
                                      char *DYND_UNUSED(data), const char *dst_arrmeta, size_t nsrc,
                                      const char *const *src_arrmeta) {
        intptr_t ckb_offset = kb.size();

        intptr_t root_ckb_offset = ckb_offset;
        kb.emplace_back<assign_from_pyobject_kernel<ndt::var_dim_type>>(kernreq);
        assign_from_pyobject_kernel<ndt::var_dim_type> *self =
            kb.get_at<assign_from_pyobject_kernel<ndt::var_dim_type>>(root_ckb_offset);
        ckb_offset = kb.size();
        self->m_offset = reinterpret_cast<const dynd::ndt::var_dim_type::metadata_type *>(dst_arrmeta)->offset;
        self->m_stride = reinterpret_cast<const dynd::ndt::var_dim_type::metadata_type *>(dst_arrmeta)->stride;
        self->m_dst_tp = dst_tp;
        self->m_dst_arrmeta = dst_arrmeta;
        const char *el_arrmeta = dst_arrmeta + sizeof(dynd::ndt::var_dim_type::metadata_type);
        kb(dynd::kernel_request_strided, nullptr, el_arrmeta, nsrc, src_arrmeta);
        ckb_offset = kb.size();
        self = kb.get_at<assign_from_pyobject_kernel<ndt::var_dim_type>>(root_ckb_offset);
        self->m_copy_dst_offset = ckb_offset - root_ckb_offset;
        // dst to dst ckernel, for broadcasting case
        //        dynd::nd::array error_mode = assign_error_fractional;
        //      kb(dynd::kernel_request_strided, el_arrmeta, 1, &el_arrmeta);
      });

      dynd::nd::assign->resolve(this, nullptr, cg, el_tp, nsrc, src_tp, nkwd, kwds, tp_vars);

      //      dynd::nd::array error_mode = assign_error_fractional;
      //  dynd::nd::assign->resolve(this, nullptr, cg, el_tp, 1, &el_tp, 1, &error_mode, tp_vars);

      return dst_tp;
    }

    /*
        void instantiate(dynd::nd::call_node *&node, char *data, dynd::nd::kernel_builder *ckb,
                         const dynd::ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                         const dynd::ndt::type *src_tp, const char *const *src_arrmeta, dynd::kernel_request_t kernreq,
                         intptr_t nkwd, const dynd::nd::array *kwds, const std::map<std::string, dynd::ndt::type>
       &tp_vars)
        {
          intptr_t ckb_offset = kb.size();
          bool dim_broadcast = false;

          intptr_t root_ckb_offset = ckb_offset;
          kb.emplace_back<assign_from_pyobject_kernel<var_dim_id>>(kernreq);
          assign_from_pyobject_kernel<var_dim_id> *self =
              kb.get_at<assign_from_pyobject_kernel<var_dim_id>>(root_ckb_offset);
          ckb_offset = kb.size();
          self->m_offset = reinterpret_cast<const dynd::ndt::var_dim_type::metadata_type *>(dst_arrmeta)->offset;
          self->m_stride = reinterpret_cast<const dynd::ndt::var_dim_type::metadata_type *>(dst_arrmeta)->stride;
          self->m_dst_tp = dst_tp;
          self->m_dst_arrmeta = dst_arrmeta;
          dynd::ndt::type el_tp = dst_tp.extended<dynd::ndt::var_dim_type>()->get_element_type();
          const char *el_arrmeta = dst_arrmeta + sizeof(dynd::ndt::var_dim_type::metadata_type);
          dynd::nd::assign->instantiate(node, NULL, ckb, el_tp, el_arrmeta, nsrc, src_tp, src_arrmeta,
                                        dynd::kernel_request_strided, nkwd, kwds, tp_vars);
          ckb_offset = kb.size();
          self = kb.get_at<assign_from_pyobject_kernel<var_dim_id>>(root_ckb_offset);
          self->m_copy_dst_offset = ckb_offset - root_ckb_offset;
          // dst to dst ckernel, for broadcasting case
          dynd::nd::array error_mode = assign_error_fractional;
          dynd::nd::assign->instantiate(node, NULL, ckb, el_tp, el_arrmeta, 1, &el_tp, &el_arrmeta,
                                        dynd::kernel_request_strided, 1, &error_mode, std::map<std::string,
       ndt::type>());
        }
    */
  };

} // namespace pydynd::nd
} // namespace pydynd

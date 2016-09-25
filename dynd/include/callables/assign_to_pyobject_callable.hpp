#pragma once

#include "kernels/assign_to_pyobject_kernel.hpp"
#include <dynd/callables/base_callable.hpp>

namespace pydynd {
namespace nd {

  template <typename Arg0Type>
  class assign_to_pyobject_callable
      : public dynd::nd::default_instantiable_callable<assign_to_pyobject_kernel<Arg0Type>> {
  public:
    assign_to_pyobject_callable()
        : dynd::nd::default_instantiable_callable<assign_to_pyobject_kernel<Arg0Type>>(
              dynd::ndt::make_type<dynd::ndt::callable_type>(dynd::ndt::make_type<pyobject_type>(),
                                                             {dynd::ndt::make_type<Arg0Type>()}))
    {
    }
  };

  template <>
  class assign_to_pyobject_callable<ndt::fixed_bytes_type> : public dynd::nd::base_callable {
  public:
    assign_to_pyobject_callable()
        : dynd::nd::base_callable(dynd::ndt::make_type<dynd::ndt::callable_type>(
              dynd::ndt::make_type<pyobject_type>(), {dynd::ndt::make_type<ndt::fixed_bytes_kind_type>()}))
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
        void instantiate(dynd::nd::call_node *&DYND_UNUSED(node), char *DYND_UNUSED(data), dynd::nd::kernel_builder
       *ckb,
                         const dynd::ndt::type &DYND_UNUSED(dst_tp), const char *DYND_UNUSED(dst_arrmeta),
                         intptr_t DYND_UNUSED(nsrc), const dynd::ndt::type *src_tp,
                         const char *const *DYND_UNUSED(src_arrmeta), dynd::kernel_request_t kernreq, intptr_t nkwd,
                         const dynd::nd::array *DYND_UNUSED(kwds),
                         const std::map<std::string, dynd::ndt::type> &DYND_UNUSED(tp_vars))
        {
          ckb->emplace_back<assign_to_pyobject_kernel<fixed_bytes_id>>(
              kernreq, src_tp[0].extended<dynd::ndt::fixed_bytes_type>()->get_data_size());
        }
    */
  };

  template <>
  class assign_to_pyobject_callable<string> : public dynd::nd::base_callable {
  public:
    assign_to_pyobject_callable()
        : dynd::nd::base_callable(dynd::ndt::make_type<dynd::ndt::callable_type>(dynd::ndt::make_type<pyobject_type>(),
                                                                                 {dynd::ndt::make_type<string>()}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), dynd::nd::call_graph &cg,
                      const dynd::ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const dynd::ndt::type *src_tp,
                      size_t DYND_UNUSED(nkwd), const dynd::nd::array *DYND_UNUSED(kwds),
                      const std::map<std::string, dynd::ndt::type> &DYND_UNUSED(tp_vars))
    {
      switch (src_tp[0].extended<dynd::ndt::base_string_type>()->get_encoding()) {
      case dynd::string_encoding_ascii:
        cg.emplace_back([](dynd::nd::kernel_builder &kb, dynd::kernel_request_t kernreq, char *DYND_UNUSED(data),
                           const char *dst_arrmeta, size_t nsrc,
                           const char *const *src_arrmeta) { kb.emplace_back<string_ascii_assign_kernel>(kernreq); });
        break;
      case dynd::string_encoding_utf_8:
        cg.emplace_back([](dynd::nd::kernel_builder &kb, dynd::kernel_request_t kernreq, char *DYND_UNUSED(data),
                           const char *dst_arrmeta, size_t nsrc,
                           const char *const *src_arrmeta) { kb.emplace_back<string_utf8_assign_kernel>(kernreq); });
        break;
      case dynd::string_encoding_ucs_2:
      case dynd::string_encoding_utf_16:
        cg.emplace_back([](dynd::nd::kernel_builder &kb, dynd::kernel_request_t kernreq, char *DYND_UNUSED(data),
                           const char *dst_arrmeta, size_t nsrc,
                           const char *const *src_arrmeta) { kb.emplace_back<string_utf16_assign_kernel>(kernreq); });
        break;
      case dynd::string_encoding_utf_32:
        cg.emplace_back([](dynd::nd::kernel_builder &kb, dynd::kernel_request_t kernreq, char *DYND_UNUSED(data),
                           const char *dst_arrmeta, size_t nsrc,
                           const char *const *src_arrmeta) { kb.emplace_back<string_utf32_assign_kernel>(kernreq); });
        break;
      default:
        throw std::runtime_error("no string_assign_kernel");
      }

      return dst_tp;
    }
  };

  template <>
  class assign_to_pyobject_callable<ndt::fixed_string_type> : public dynd::nd::base_callable {
  public:
    assign_to_pyobject_callable()
        : dynd::nd::base_callable(dynd::ndt::make_type<dynd::ndt::callable_type>(
              dynd::ndt::make_type<pyobject_type>(), {dynd::ndt::make_type<ndt::fixed_string_kind_type>()}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), dynd::nd::call_graph &cg,
                      const dynd::ndt::type &dst_tp, size_t DYND_UNUSED(nsrc), const dynd::ndt::type *src_tp,
                      size_t DYND_UNUSED(nkwd), const dynd::nd::array *DYND_UNUSED(kwds),
                      const std::map<std::string, dynd::ndt::type> &DYND_UNUSED(tp_vars))
    {
      size_t src0_data_size = src_tp[0].get_data_size();
      switch (src_tp[0].extended<dynd::ndt::base_string_type>()->get_encoding()) {
      case dynd::string_encoding_ascii:
        cg.emplace_back([src0_data_size](dynd::nd::kernel_builder &kb, dynd::kernel_request_t kernreq,
                                         char *DYND_UNUSED(data), const char *DYND_UNUSED(dst_arrmeta),
                                         size_t DYND_UNUSED(nsrc), const char *const *DYND_UNUSED(src_arrmeta)) {
          kb.emplace_back<fixed_string_ascii_assign_kernel>(kernreq, src0_data_size);
        });
        break;
      case dynd::string_encoding_utf_8:
        cg.emplace_back([src0_data_size](dynd::nd::kernel_builder &kb, dynd::kernel_request_t kernreq,
                                         char *DYND_UNUSED(data), const char *DYND_UNUSED(dst_arrmeta),
                                         size_t DYND_UNUSED(nsrc), const char *const *DYND_UNUSED(src_arrmeta)) {
          kb.emplace_back<fixed_string_utf8_assign_kernel>(kernreq, src0_data_size);
        });
        break;
      case dynd::string_encoding_ucs_2:
      case dynd::string_encoding_utf_16:
        cg.emplace_back([src0_data_size](dynd::nd::kernel_builder &kb, dynd::kernel_request_t kernreq,
                                         char *DYND_UNUSED(data), const char *DYND_UNUSED(dst_arrmeta),
                                         size_t DYND_UNUSED(nsrc), const char *const *DYND_UNUSED(src_arrmeta)) {
          kb.emplace_back<fixed_string_utf16_assign_kernel>(kernreq, src0_data_size);
        });
        break;
      case dynd::string_encoding_utf_32:
        cg.emplace_back([src0_data_size](dynd::nd::kernel_builder &kb, dynd::kernel_request_t kernreq,
                                         char *DYND_UNUSED(data), const char *DYND_UNUSED(dst_arrmeta),
                                         size_t DYND_UNUSED(nsrc), const char *const *DYND_UNUSED(src_arrmeta)) {
          kb.emplace_back<fixed_string_utf32_assign_kernel>(kernreq, src0_data_size);
        });
        break;
      default:
        throw std::runtime_error("no fixed_string_assign_kernel");
      }

      return dst_tp;
    }
  };

  template <>
  class assign_to_pyobject_callable<ndt::option_type> : public dynd::nd::base_callable {
  public:
    assign_to_pyobject_callable()
        : dynd::nd::base_callable(dynd::ndt::make_type<dynd::ndt::callable_type>(
              dynd::ndt::make_type<pyobject_type>(),
              {dynd::ndt::make_type<ndt::option_type>(ndt::make_type<ndt::any_kind_type>())}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), dynd::nd::call_graph &cg,
                      const dynd::ndt::type &dst_tp, size_t nsrc, const dynd::ndt::type *src_tp,
                      size_t DYND_UNUSED(nkwd), const dynd::nd::array *DYND_UNUSED(kwds),
                      const std::map<std::string, dynd::ndt::type> &tp_vars)
    {
      dynd::ndt::type src_value_tp = src_tp[0].extended<dynd::ndt::option_type>()->get_value_type();
      cg.emplace_back([](dynd::nd::kernel_builder &kb, dynd::kernel_request_t kernreq, char *DYND_UNUSED(data),
                         const char *dst_arrmeta, size_t nsrc, const char *const *src_arrmeta) {
        intptr_t root_ckb_offset = kb.size();
        kb.emplace_back<assign_to_pyobject_kernel<ndt::option_type>>(kernreq);

        assign_to_pyobject_kernel<ndt::option_type> *self_ck =
            kb.get_at<assign_to_pyobject_kernel<ndt::option_type>>(root_ckb_offset);
        kb(dynd::kernel_request_single, nullptr, nullptr, nsrc, src_arrmeta);

        self_ck = kb.get_at<assign_to_pyobject_kernel<ndt::option_type>>(root_ckb_offset);
        self_ck->m_assign_value_offset = kb.size() - root_ckb_offset;

        kb(dynd::kernel_request_single, nullptr, dst_arrmeta, nsrc, src_arrmeta);
      });

      dynd::nd::is_na->resolve(this, nullptr, cg, dynd::ndt::make_type<dynd::bool1>(), nsrc, src_tp, 0, nullptr,
                               tp_vars);
      dynd::nd::assign->resolve(this, nullptr, cg, dst_tp, nsrc, &src_value_tp, 0, NULL, tp_vars);

      return dst_tp;
    }

    /*
        void instantiate(dynd::nd::call_node *&node, char *data, dynd::nd::kernel_builder *ckb,
                         const dynd::ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                         const dynd::ndt::type *src_tp, const char *const *src_arrmeta, dynd::kernel_request_t kernreq,
                         intptr_t nkwd, const dynd::nd::array *kwds, const std::map<std::string, dynd::ndt::type>
       &tp_vars)
        {
          intptr_t root_ckb_offset = ckb->size();
          ckb->emplace_back<assign_to_pyobject_kernel<option_id>>(kernreq);
          assign_to_pyobject_kernel<option_id> *self_ck =
              ckb->get_at<assign_to_pyobject_kernel<option_id>>(root_ckb_offset);
          dynd::nd::is_na->instantiate(node, NULL, ckb, dynd::ndt::make_type<dynd::bool1>(), NULL, nsrc, src_tp,
                                       src_arrmeta, dynd::kernel_request_single, 0, NULL, tp_vars);
          self_ck = ckb->get_at<assign_to_pyobject_kernel<option_id>>(root_ckb_offset);
          self_ck->m_assign_value_offset = ckb->size() - root_ckb_offset;
          dynd::ndt::type src_value_tp = src_tp[0].extended<dynd::ndt::option_type>()->get_value_type();
          dynd::nd::assign->instantiate(node, NULL, ckb, dst_tp, dst_arrmeta, nsrc, &src_value_tp, src_arrmeta,
                                        dynd::kernel_request_single, 0, NULL, tp_vars);
        }
    */
  };

  template <>
  class assign_to_pyobject_callable<ndt::tuple_type> : public dynd::nd::base_callable {
  public:
    assign_to_pyobject_callable()
        : dynd::nd::base_callable(dynd::ndt::make_type<dynd::ndt::callable_type>(
              dynd::ndt::make_type<pyobject_type>(), {dynd::ndt::make_type<ndt::tuple_type>(true)}))
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
        void instantiate(dynd::nd::call_node *&node, char *data, dynd::nd::kernel_builder *ckb,
                         const dynd::ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                         const dynd::ndt::type *src_tp, const char *const *src_arrmeta, dynd::kernel_request_t kernreq,
                         intptr_t nkwd, const dynd::nd::array *kwds, const std::map<std::string, dynd::ndt::type>
       &tp_vars)
        {
          intptr_t ckb_offset = ckb->size();
          intptr_t root_ckb_offset = ckb_offset;
          ckb->emplace_back<assign_to_pyobject_kernel<tuple_id>>(kernreq, src_tp[0], src_arrmeta[0]);
          assign_to_pyobject_kernel<tuple_id> *self_ck =
       ckb->get_at<assign_to_pyobject_kernel<tuple_id>>(root_ckb_offset);
          ckb_offset = ckb->size();
          intptr_t field_count = src_tp[0].extended<dynd::ndt::tuple_type>()->get_field_count();
          const dynd::ndt::type *field_types = src_tp[0].extended<dynd::ndt::tuple_type>()->get_field_types_raw();
          const uintptr_t *arrmeta_offsets = src_tp[0].extended<dynd::ndt::tuple_type>()->get_arrmeta_offsets_raw();
          self_ck->m_copy_el_offsets.resize(field_count);
          for (intptr_t i = 0; i < field_count; ++i) {
            ckb->reserve(ckb_offset);
            self_ck = ckb->get_at<assign_to_pyobject_kernel<tuple_id>>(root_ckb_offset);
            self_ck->m_copy_el_offsets[i] = ckb_offset - root_ckb_offset;
            const char *field_arrmeta = src_arrmeta[0] + arrmeta_offsets[i];
            dynd::nd::assign->instantiate(node, data, ckb, dst_tp, dst_arrmeta, nsrc, &field_types[i], &field_arrmeta,
                                          dynd::kernel_request_single, nkwd, kwds, tp_vars);
            ckb_offset = ckb->size();
          }
        }
    */
  };

  template <>
  class assign_to_pyobject_callable<ndt::struct_type> : public dynd::nd::base_callable {
  public:
    assign_to_pyobject_callable()
        : dynd::nd::base_callable(dynd::ndt::make_type<dynd::ndt::callable_type>(
              dynd::ndt::make_type<pyobject_type>(), {dynd::ndt::make_type<ndt::struct_type>(true)}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), dynd::nd::call_graph &cg,
                      const dynd::ndt::type &dst_tp, size_t nsrc, const dynd::ndt::type *src_tp,
                      size_t DYND_UNUSED(nkwd), const dynd::nd::array *DYND_UNUSED(kwds),
                      const std::map<std::string, dynd::ndt::type> &tp_vars)
    {
      intptr_t field_count = src_tp[0].extended<dynd::ndt::struct_type>()->get_field_count();
      const dynd::ndt::type *field_types = src_tp[0].extended<dynd::ndt::struct_type>()->get_field_types_raw();
      const std::vector<uintptr_t> &arrmeta_offsets =
          src_tp[0].extended<dynd::ndt::struct_type>()->get_arrmeta_offsets();

      ndt::type src0_tp = src_tp[0];
      cg.emplace_back([src0_tp, arrmeta_offsets, field_count](
          dynd::nd::kernel_builder &kb, dynd::kernel_request_t kernreq, char *DYND_UNUSED(data),
          const char *dst_arrmeta, size_t nsrc, const char *const *src_arrmeta) {
        intptr_t ckb_offset = kb.size();
        intptr_t root_ckb_offset = ckb_offset;
        kb.emplace_back<assign_to_pyobject_kernel<ndt::struct_type>>(kernreq);

        assign_to_pyobject_kernel<ndt::struct_type> *self_ck =
            kb.get_at<assign_to_pyobject_kernel<ndt::struct_type>>(root_ckb_offset);

        ckb_offset = kb.size();
        self_ck->m_src_tp = src0_tp;
        self_ck->m_src_arrmeta = src_arrmeta[0];
        self_ck->m_field_names = capture_if_not_null(PyTuple_New(field_count));
        for (intptr_t i = 0; i < field_count; ++i) {
          const dynd::string &rawname = src0_tp.extended<dynd::ndt::struct_type>()->get_field_name(i);
          pydynd::py_ref name =
              capture_if_not_null(PyUnicode_DecodeUTF8(rawname.begin(), rawname.end() - rawname.begin(), NULL));
          PyTuple_SET_ITEM(self_ck->m_field_names.get(), i, pydynd::release(std::move(name)));
        }
        self_ck->m_copy_el_offsets.resize(field_count);

        for (intptr_t i = 0; i < field_count; ++i) {
          kb.reserve(ckb_offset);
          self_ck = kb.get_at<assign_to_pyobject_kernel<ndt::struct_type>>(root_ckb_offset);
          self_ck->m_copy_el_offsets[i] = ckb_offset - root_ckb_offset;
          const char *field_arrmeta = src_arrmeta[0] + arrmeta_offsets[i];
          kb(dynd::kernel_request_single, nullptr, dst_arrmeta, nsrc, &field_arrmeta);
          ckb_offset = kb.size();
        }

      });

      for (intptr_t i = 0; i < field_count; ++i) {
        dynd::nd::assign->resolve(this, nullptr, cg, dst_tp, nsrc, &field_types[i], 0, nullptr, tp_vars);
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
          intptr_t ckb_offset = ckb->size();
          intptr_t root_ckb_offset = ckb_offset;
          ckb->emplace_back<assign_to_pyobject_kernel<struct_id>>(kernreq);
          assign_to_pyobject_kernel<struct_id> *self_ck =
              ckb->get_at<assign_to_pyobject_kernel<struct_id>>(root_ckb_offset);
          ckb_offset = ckb->size();
          self_ck->m_src_tp = src_tp[0];
          self_ck->m_src_arrmeta = src_arrmeta[0];
          intptr_t field_count = src_tp[0].extended<dynd::ndt::struct_type>()->get_field_count();
          const dynd::ndt::type *field_types = src_tp[0].extended<dynd::ndt::struct_type>()->get_field_types_raw();
          const uintptr_t *arrmeta_offsets = src_tp[0].extended<dynd::ndt::struct_type>()->get_arrmeta_offsets_raw();
          self_ck->m_field_names = capture_if_not_null(PyTuple_New(field_count));
          for (intptr_t i = 0; i < field_count; ++i) {
            const dynd::string &rawname = src_tp[0].extended<dynd::ndt::struct_type>()->get_field_name(i);
            pydynd::py_ref name =
                capture_if_not_null(PyUnicode_DecodeUTF8(rawname.begin(), rawname.end() - rawname.begin(), NULL));
            PyTuple_SET_ITEM(self_ck->m_field_names.get(), i, pydynd::release(std::move(name)));
          }
          self_ck->m_copy_el_offsets.resize(field_count);
          for (intptr_t i = 0; i < field_count; ++i) {
            ckb->reserve(ckb_offset);
            self_ck = ckb->get_at<assign_to_pyobject_kernel<struct_id>>(root_ckb_offset);
            self_ck->m_copy_el_offsets[i] = ckb_offset - root_ckb_offset;
            const char *field_arrmeta = src_arrmeta[0] + arrmeta_offsets[i];
            dynd::nd::assign->instantiate(node, NULL, ckb, dst_tp, dst_arrmeta, nsrc, &field_types[i], &field_arrmeta,
                                          dynd::kernel_request_single, 0, NULL, tp_vars);
            ckb_offset = ckb->size();
          }
        }
    */
  };

  template <>
  class assign_to_pyobject_callable<ndt::fixed_dim_type> : public dynd::nd::base_callable {
  public:
    assign_to_pyobject_callable()
        : dynd::nd::base_callable(dynd::ndt::make_type<dynd::ndt::callable_type>(dynd::ndt::make_type<pyobject_type>(),
                                                                                 {dynd::ndt::type("Fixed * Any")}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), dynd::nd::call_graph &cg,
                      const dynd::ndt::type &dst_tp, size_t nsrc, const dynd::ndt::type *src_tp, size_t nkwd,
                      const dynd::nd::array *kwds, const std::map<std::string, dynd::ndt::type> &tp_vars)
    {
      cg.emplace_back([](dynd::nd::kernel_builder &kb, dynd::kernel_request_t kernreq, char *DYND_UNUSED(data),
                         const char *dst_arrmeta, size_t DYND_UNUSED(nsrc), const char *const *src_arrmeta) {
        const char *src_element_arrmeta[1] = {src_arrmeta[0] + sizeof(size_stride_t)};
        kb.emplace_back<assign_to_pyobject_kernel<ndt::fixed_dim_type>>(
            kernreq, reinterpret_cast<const size_stride_t *>(src_arrmeta[0])->dim_size,
            reinterpret_cast<const size_stride_t *>(src_arrmeta[0])->stride);

        kb(dynd::kernel_request_strided, nullptr, dst_arrmeta, 1, src_element_arrmeta);
      });

      dynd::ndt::type src_element_tp[1] = {src_tp[0].extended<ndt::fixed_dim_type>()->get_element_type()};
      dynd::nd::assign->resolve(this, nullptr, cg, dst_tp, nsrc, src_element_tp, nkwd, kwds, tp_vars);

      return dst_tp;
    }

    /*
        void instantiate(dynd::nd::call_node *node, char *data, dynd::nd::kernel_builder *ckb,
                         const dynd::ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                         const dynd::ndt::type *src_tp, const char *const *src_arrmeta, dynd::kernel_request_t kernreq,
                         intptr_t nkwd, const dynd::nd::array *kwds, const std::map<std::string, dynd::ndt::type>
       &tp_vars)
        {
          intptr_t ckb_offset = ckb->size();
          intptr_t dim_size, stride;
          dynd::ndt::type el_tp;
          const char *el_arrmeta;
          if (src_tp[0].get_as_strided(src_arrmeta[0], &dim_size, &stride, &el_tp, &el_arrmeta)) {
            ckb->emplace_back<assign_to_pyobject_kernel<fixed_dim_id>>(kernreq, dim_size, stride);
            ckb_offset = ckb->size();
            dynd::nd::assign->instantiate(node, data, ckb, dst_tp, dst_arrmeta, nsrc, &el_tp, &el_arrmeta,
                                          dynd::kernel_request_strided, nkwd, kwds, tp_vars);
            return;
          }

          throw std::runtime_error("cannot process as strided");
        }
    */
  };

  template <>
  class assign_to_pyobject_callable<ndt::var_dim_type> : public dynd::nd::base_callable {
  public:
    assign_to_pyobject_callable()
        : dynd::nd::base_callable(dynd::ndt::make_type<dynd::ndt::callable_type>(
              dynd::ndt::make_type<pyobject_type>(),
              {dynd::ndt::make_type<ndt::var_dim_type>(ndt::make_type<ndt::any_kind_type>())}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), dynd::nd::call_graph &cg,
                      const dynd::ndt::type &dst_tp, size_t nsrc, const dynd::ndt::type *src_tp, size_t nkwd,
                      const dynd::nd::array *kwds, const std::map<std::string, dynd::ndt::type> &tp_vars)
    {
      dynd::ndt::type el_tp = src_tp[0].extended<dynd::ndt::var_dim_type>()->get_element_type();
      cg.emplace_back([el_tp](dynd::nd::kernel_builder &kb, dynd::kernel_request_t kernreq, char *DYND_UNUSED(data),
                              const char *dst_arrmeta, size_t nsrc, const char *const *src_arrmeta) {
        intptr_t ckb_offset = kb.size();
        kb.emplace_back<assign_to_pyobject_kernel<ndt::var_dim_type>>(
            kernreq, reinterpret_cast<const dynd::ndt::var_dim_type::metadata_type *>(src_arrmeta[0])->offset,
            reinterpret_cast<const dynd::ndt::var_dim_type::metadata_type *>(src_arrmeta[0])->stride);

        ckb_offset = kb.size();
        const char *el_arrmeta = src_arrmeta[0] + sizeof(dynd::ndt::var_dim_type::metadata_type);
        kb(dynd::kernel_request_strided, nullptr, dst_arrmeta, nsrc, &el_arrmeta);
      });

      dynd::nd::assign->resolve(this, nullptr, cg, dst_tp, nsrc, &el_tp, nkwd, kwds, tp_vars);

      return dst_tp;
    }

    /*
        void instantiate(dynd::nd::call_node *node, char *data, dynd::nd::kernel_builder *ckb,
                         const dynd::ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
                         const dynd::ndt::type *src_tp, const char *const *src_arrmeta, dynd::kernel_request_t kernreq,
                         intptr_t nkwd, const dynd::nd::array *kwds, const std::map<std::string, dynd::ndt::type>
       &tp_vars)
        {
          intptr_t ckb_offset = ckb->size();
          ckb->emplace_back<assign_to_pyobject_kernel<var_dim_id>>(
              kernreq, reinterpret_cast<const dynd::ndt::var_dim_type::metadata_type *>(src_arrmeta[0])->offset,
              reinterpret_cast<const dynd::ndt::var_dim_type::metadata_type *>(src_arrmeta[0])->stride);
          ckb_offset = ckb->size();
          dynd::ndt::type el_tp = src_tp[0].extended<dynd::ndt::var_dim_type>()->get_element_type();
          const char *el_arrmeta = src_arrmeta[0] + sizeof(dynd::ndt::var_dim_type::metadata_type);
          dynd::nd::assign->instantiate(node, data, ckb, dst_tp, dst_arrmeta, nsrc, &el_tp, &el_arrmeta,
                                        dynd::kernel_request_strided, nkwd, kwds, tp_vars);
        }
    */
  };

} // namespace pydynd::nd
} // namespace pydynd

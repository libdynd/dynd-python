#pragma once

#include <dynd/callables/base_callable.hpp>
#include "kernels/assign_to_pyobject_kernel.hpp"

namespace pydynd {
namespace nd {

  template <type_id_t Arg0ID>
  class assign_to_pyobject_callable
      : public dynd::nd::default_instantiable_callable<assign_to_pyobject_kernel<Arg0ID>> {
  public:
    assign_to_pyobject_callable()
        : dynd::nd::default_instantiable_callable<assign_to_pyobject_kernel<Arg0ID>>(
              dynd::ndt::callable_type::make(dynd::ndt::make_type<pyobject_type>(), {dynd::ndt::type(Arg0ID)}))
    {
    }
  };

  template <>
  class assign_to_pyobject_callable<fixed_bytes_id> : public dynd::nd::base_callable {
  public:
    assign_to_pyobject_callable()
        : dynd::nd::base_callable(
              dynd::ndt::callable_type::make(dynd::ndt::make_type<pyobject_type>(), {dynd::ndt::type(fixed_bytes_id)}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), dynd::nd::call_graph &cg,
                      const dynd::ndt::type &dst_tp, size_t DYND_UNUSED(nsrc),
                      const dynd::ndt::type *DYND_UNUSED(src_tp), size_t DYND_UNUSED(nkwd),
                      const dynd::nd::array *DYND_UNUSED(kwds),
                      const std::map<std::string, dynd::ndt::type> &DYND_UNUSED(tp_vars))
    {
      cg.emplace_back(this);
      return dst_tp;
    }

    void instantiate(char *DYND_UNUSED(data), dynd::nd::kernel_builder *ckb, const dynd::ndt::type &DYND_UNUSED(dst_tp),
                     const char *DYND_UNUSED(dst_arrmeta), intptr_t DYND_UNUSED(nsrc), const dynd::ndt::type *src_tp,
                     const char *const *DYND_UNUSED(src_arrmeta), dynd::kernel_request_t kernreq, intptr_t nkwd,
                     const dynd::nd::array *DYND_UNUSED(kwds),
                     const std::map<std::string, dynd::ndt::type> &DYND_UNUSED(tp_vars))
    {
      ckb->emplace_back<assign_to_pyobject_kernel<fixed_bytes_id>>(
          kernreq, src_tp[0].extended<dynd::ndt::fixed_bytes_type>()->get_data_size());
    }
  };

  template <>
  class assign_to_pyobject_callable<string_id> : public dynd::nd::base_callable {
  public:
    assign_to_pyobject_callable()
        : dynd::nd::base_callable(
              dynd::ndt::callable_type::make(dynd::ndt::make_type<pyobject_type>(), {dynd::ndt::type(string_id)}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), dynd::nd::call_graph &cg,
                      const dynd::ndt::type &dst_tp, size_t DYND_UNUSED(nsrc),
                      const dynd::ndt::type *DYND_UNUSED(src_tp), size_t DYND_UNUSED(nkwd),
                      const dynd::nd::array *DYND_UNUSED(kwds),
                      const std::map<std::string, dynd::ndt::type> &DYND_UNUSED(tp_vars))
    {
      cg.emplace_back(this);
      return dst_tp;
    }

    void instantiate(char *data, dynd::nd::kernel_builder *ckb, const dynd::ndt::type &dst_tp, const char *dst_arrmeta,
                     intptr_t nsrc, const dynd::ndt::type *src_tp, const char *const *src_arrmeta,
                     dynd::kernel_request_t kernreq, intptr_t nkwd, const dynd::nd::array *kwds,
                     const std::map<std::string, dynd::ndt::type> &tp_vars)
    {
      switch (src_tp[0].extended<dynd::ndt::base_string_type>()->get_encoding()) {
      case dynd::string_encoding_ascii:
        ckb->emplace_back<::detail::string_ascii_assign_kernel>(kernreq);
        break;
      case dynd::string_encoding_utf_8:
        ckb->emplace_back<::detail::string_utf8_assign_kernel>(kernreq);
        break;
      case dynd::string_encoding_ucs_2:
      case dynd::string_encoding_utf_16:
        ckb->emplace_back<::detail::string_utf16_assign_kernel>(kernreq);
        break;
      case dynd::string_encoding_utf_32:
        ckb->emplace_back<::detail::string_utf32_assign_kernel>(kernreq);
        break;
      default:
        throw std::runtime_error("no string_assign_kernel");
      }
    }
  };

  template <>
  class assign_to_pyobject_callable<fixed_string_id> : public dynd::nd::base_callable {
  public:
    assign_to_pyobject_callable()
        : dynd::nd::base_callable(
              dynd::ndt::callable_type::make(dynd::ndt::make_type<pyobject_type>(), {dynd::ndt::type(fixed_string_id)}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), dynd::nd::call_graph &cg,
                      const dynd::ndt::type &dst_tp, size_t DYND_UNUSED(nsrc),
                      const dynd::ndt::type *DYND_UNUSED(src_tp), size_t DYND_UNUSED(nkwd),
                      const dynd::nd::array *DYND_UNUSED(kwds),
                      const std::map<std::string, dynd::ndt::type> &DYND_UNUSED(tp_vars))
    {
      cg.emplace_back(this);
      return dst_tp;
    }

    void instantiate(char *data, dynd::nd::kernel_builder *ckb, const dynd::ndt::type &dst_tp, const char *dst_arrmeta,
                     intptr_t nsrc, const dynd::ndt::type *src_tp, const char *const *src_arrmeta,
                     dynd::kernel_request_t kernreq, intptr_t nkwd, const dynd::nd::array *kwds,
                     const std::map<std::string, dynd::ndt::type> &tp_vars)
    {
      switch (src_tp[0].extended<dynd::ndt::base_string_type>()->get_encoding()) {
      case dynd::string_encoding_ascii:
        ckb->emplace_back<::detail::fixed_string_ascii_assign_kernel>(kernreq, src_tp[0].get_data_size());
        break;
      case dynd::string_encoding_utf_8:
        ckb->emplace_back<::detail::fixed_string_utf8_assign_kernel>(kernreq, src_tp[0].get_data_size());
        break;
      case dynd::string_encoding_ucs_2:
      case dynd::string_encoding_utf_16:
        ckb->emplace_back<::detail::fixed_string_utf16_assign_kernel>(kernreq, src_tp[0].get_data_size());
        break;
      case dynd::string_encoding_utf_32:
        ckb->emplace_back<::detail::fixed_string_utf32_assign_kernel>(kernreq, src_tp[0].get_data_size());
        break;
      default:
        throw std::runtime_error("no fixed_string_assign_kernel");
      }
    }
  };

  template <>
  class assign_to_pyobject_callable<option_id> : public dynd::nd::base_callable {
  public:
    assign_to_pyobject_callable()
        : dynd::nd::base_callable(
              dynd::ndt::callable_type::make(dynd::ndt::make_type<pyobject_type>(), {dynd::ndt::type(option_id)}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), dynd::nd::call_graph &cg,
                      const dynd::ndt::type &dst_tp, size_t DYND_UNUSED(nsrc),
                      const dynd::ndt::type *DYND_UNUSED(src_tp), size_t DYND_UNUSED(nkwd),
                      const dynd::nd::array *DYND_UNUSED(kwds),
                      const std::map<std::string, dynd::ndt::type> &DYND_UNUSED(tp_vars))
    {
      cg.emplace_back(this);
      return dst_tp;
    }

    void instantiate(char *data, dynd::nd::kernel_builder *ckb, const dynd::ndt::type &dst_tp, const char *dst_arrmeta,
                     intptr_t nsrc, const dynd::ndt::type *src_tp, const char *const *src_arrmeta,
                     dynd::kernel_request_t kernreq, intptr_t nkwd, const dynd::nd::array *kwds,
                     const std::map<std::string, dynd::ndt::type> &tp_vars)
    {
      intptr_t root_ckb_offset = ckb->size();
      ckb->emplace_back<assign_to_pyobject_kernel<option_id>>(kernreq);
      assign_to_pyobject_kernel<option_id> *self_ck =
          ckb->get_at<assign_to_pyobject_kernel<option_id>>(root_ckb_offset);
      dynd::nd::is_na->instantiate(NULL, ckb, dynd::ndt::make_type<dynd::bool1>(), NULL, nsrc, src_tp, src_arrmeta,
                                   dynd::kernel_request_single, 0, NULL, tp_vars);
      self_ck = ckb->get_at<assign_to_pyobject_kernel<option_id>>(root_ckb_offset);
      self_ck->m_assign_value_offset = ckb->size() - root_ckb_offset;
      dynd::ndt::type src_value_tp = src_tp[0].extended<dynd::ndt::option_type>()->get_value_type();
      dynd::nd::assign->instantiate(NULL, ckb, dst_tp, dst_arrmeta, nsrc, &src_value_tp, src_arrmeta,
                                    dynd::kernel_request_single, 0, NULL, tp_vars);
    }
  };

  template <>
  class assign_to_pyobject_callable<tuple_id> : public dynd::nd::base_callable {
  public:
    assign_to_pyobject_callable()
        : dynd::nd::base_callable(
              dynd::ndt::callable_type::make(dynd::ndt::make_type<pyobject_type>(), {dynd::ndt::type(tuple_id)}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), dynd::nd::call_graph &cg,
                      const dynd::ndt::type &dst_tp, size_t DYND_UNUSED(nsrc),
                      const dynd::ndt::type *DYND_UNUSED(src_tp), size_t DYND_UNUSED(nkwd),
                      const dynd::nd::array *DYND_UNUSED(kwds),
                      const std::map<std::string, dynd::ndt::type> &DYND_UNUSED(tp_vars))
    {
      cg.emplace_back(this);
      return dst_tp;
    }

    void instantiate(char *data, dynd::nd::kernel_builder *ckb, const dynd::ndt::type &dst_tp, const char *dst_arrmeta,
                     intptr_t nsrc, const dynd::ndt::type *src_tp, const char *const *src_arrmeta,
                     dynd::kernel_request_t kernreq, intptr_t nkwd, const dynd::nd::array *kwds,
                     const std::map<std::string, dynd::ndt::type> &tp_vars)
    {
      intptr_t ckb_offset = ckb->size();
      intptr_t root_ckb_offset = ckb_offset;
      ckb->emplace_back<assign_to_pyobject_kernel<tuple_id>>(kernreq, src_tp[0], src_arrmeta[0]);
      assign_to_pyobject_kernel<tuple_id> *self_ck = ckb->get_at<assign_to_pyobject_kernel<tuple_id>>(root_ckb_offset);
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
        dynd::nd::assign->instantiate(data, ckb, dst_tp, dst_arrmeta, nsrc, &field_types[i], &field_arrmeta,
                                      dynd::kernel_request_single, nkwd, kwds, tp_vars);
        ckb_offset = ckb->size();
      }
    }
  };

  template <>
  class assign_to_pyobject_callable<struct_id> : public dynd::nd::base_callable {
  public:
    assign_to_pyobject_callable()
        : dynd::nd::base_callable(
              dynd::ndt::callable_type::make(dynd::ndt::make_type<pyobject_type>(), {dynd::ndt::type(struct_id)}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), dynd::nd::call_graph &cg,
                      const dynd::ndt::type &dst_tp, size_t DYND_UNUSED(nsrc),
                      const dynd::ndt::type *DYND_UNUSED(src_tp), size_t DYND_UNUSED(nkwd),
                      const dynd::nd::array *DYND_UNUSED(kwds),
                      const std::map<std::string, dynd::ndt::type> &DYND_UNUSED(tp_vars))
    {
      cg.emplace_back(this);
      return dst_tp;
    }

    void instantiate(char *data, dynd::nd::kernel_builder *ckb, const dynd::ndt::type &dst_tp, const char *dst_arrmeta,
                     intptr_t nsrc, const dynd::ndt::type *src_tp, const char *const *src_arrmeta,
                     dynd::kernel_request_t kernreq, intptr_t nkwd, const dynd::nd::array *kwds,
                     const std::map<std::string, dynd::ndt::type> &tp_vars)
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
      self_ck->m_field_names.reset(PyTuple_New(field_count));
      for (intptr_t i = 0; i < field_count; ++i) {
        const dynd::string &rawname = src_tp[0].extended<dynd::ndt::struct_type>()->get_field_name(i);
        pydynd::pyobject_ownref name(PyUnicode_DecodeUTF8(rawname.begin(), rawname.end() - rawname.begin(), NULL));
        PyTuple_SET_ITEM(self_ck->m_field_names.get(), i, name.release());
      }
      self_ck->m_copy_el_offsets.resize(field_count);
      for (intptr_t i = 0; i < field_count; ++i) {
        ckb->reserve(ckb_offset);
        self_ck = ckb->get_at<assign_to_pyobject_kernel<struct_id>>(root_ckb_offset);
        self_ck->m_copy_el_offsets[i] = ckb_offset - root_ckb_offset;
        const char *field_arrmeta = src_arrmeta[0] + arrmeta_offsets[i];
        dynd::nd::assign->instantiate(NULL, ckb, dst_tp, dst_arrmeta, nsrc, &field_types[i], &field_arrmeta,
                                      dynd::kernel_request_single, 0, NULL, tp_vars);
        ckb_offset = ckb->size();
      }
    }
  };

  template <>
  class assign_to_pyobject_callable<fixed_dim_id> : public dynd::nd::base_callable {
  public:
    assign_to_pyobject_callable()
        : dynd::nd::base_callable(
              dynd::ndt::callable_type::make(dynd::ndt::make_type<pyobject_type>(), {dynd::ndt::type(fixed_dim_id)}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), dynd::nd::call_graph &cg,
                      const dynd::ndt::type &dst_tp, size_t DYND_UNUSED(nsrc),
                      const dynd::ndt::type *DYND_UNUSED(src_tp), size_t DYND_UNUSED(nkwd),
                      const dynd::nd::array *DYND_UNUSED(kwds),
                      const std::map<std::string, dynd::ndt::type> &DYND_UNUSED(tp_vars))
    {
      cg.emplace_back(this);
      return dst_tp;
    }

    void instantiate(char *data, dynd::nd::kernel_builder *ckb, const dynd::ndt::type &dst_tp, const char *dst_arrmeta,
                     intptr_t nsrc, const dynd::ndt::type *src_tp, const char *const *src_arrmeta,
                     dynd::kernel_request_t kernreq, intptr_t nkwd, const dynd::nd::array *kwds,
                     const std::map<std::string, dynd::ndt::type> &tp_vars)
    {
      intptr_t ckb_offset = ckb->size();
      intptr_t dim_size, stride;
      dynd::ndt::type el_tp;
      const char *el_arrmeta;
      if (src_tp[0].get_as_strided(src_arrmeta[0], &dim_size, &stride, &el_tp, &el_arrmeta)) {
        ckb->emplace_back<assign_to_pyobject_kernel<fixed_dim_id>>(kernreq, dim_size, stride);
        ckb_offset = ckb->size();
        dynd::nd::assign->instantiate(data, ckb, dst_tp, dst_arrmeta, nsrc, &el_tp, &el_arrmeta,
                                      dynd::kernel_request_strided, nkwd, kwds, tp_vars);
        return;
      }

      throw std::runtime_error("cannot process as strided");
    }
  };

  template <>
  class assign_to_pyobject_callable<var_dim_id> : public dynd::nd::base_callable {
  public:
    assign_to_pyobject_callable()
        : dynd::nd::base_callable(
              dynd::ndt::callable_type::make(dynd::ndt::make_type<pyobject_type>(), {dynd::ndt::type(var_dim_id)}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data), dynd::nd::call_graph &cg,
                      const dynd::ndt::type &dst_tp, size_t DYND_UNUSED(nsrc),
                      const dynd::ndt::type *DYND_UNUSED(src_tp), size_t DYND_UNUSED(nkwd),
                      const dynd::nd::array *DYND_UNUSED(kwds),
                      const std::map<std::string, dynd::ndt::type> &DYND_UNUSED(tp_vars))
    {
      cg.emplace_back(this);
      return dst_tp;
    }

    void instantiate(char *data, dynd::nd::kernel_builder *ckb, const dynd::ndt::type &dst_tp, const char *dst_arrmeta,
                     intptr_t nsrc, const dynd::ndt::type *src_tp, const char *const *src_arrmeta,
                     dynd::kernel_request_t kernreq, intptr_t nkwd, const dynd::nd::array *kwds,
                     const std::map<std::string, dynd::ndt::type> &tp_vars)
    {
      intptr_t ckb_offset = ckb->size();
      ckb->emplace_back<assign_to_pyobject_kernel<var_dim_id>>(
          kernreq, reinterpret_cast<const dynd::ndt::var_dim_type::metadata_type *>(src_arrmeta[0])->offset,
          reinterpret_cast<const dynd::ndt::var_dim_type::metadata_type *>(src_arrmeta[0])->stride);
      ckb_offset = ckb->size();
      dynd::ndt::type el_tp = src_tp[0].extended<dynd::ndt::var_dim_type>()->get_element_type();
      const char *el_arrmeta = src_arrmeta[0] + sizeof(dynd::ndt::var_dim_type::metadata_type);
      dynd::nd::assign->instantiate(data, ckb, dst_tp, dst_arrmeta, nsrc, &el_tp, &el_arrmeta,
                                    dynd::kernel_request_strided, nkwd, kwds, tp_vars);
    }
  };

} // namespace pydynd::nd
} // namespace pydynd

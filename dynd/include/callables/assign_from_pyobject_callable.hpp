#pragma once

#include <dynd/callables/base_callable.hpp>
#include "kernels/assign_from_pyobject_kernel.hpp"

namespace pydynd {
namespace nd {

  template <type_id_t ResID>
  class assign_from_pyobject_callable
      : public dynd::nd::default_instantiable_callable<assign_from_pyobject_kernel<ResID>> {
  public:
    assign_from_pyobject_callable()
        : dynd::nd::default_instantiable_callable<assign_from_pyobject_kernel<ResID>>(
              dynd::ndt::callable_type::make(ResID, {dynd::ndt::make_type<pyobject_type>()}))
    {
    }
  };

  template <>
  class assign_from_pyobject_callable<bytes_id> : public dynd::nd::base_callable {
  public:
    assign_from_pyobject_callable()
        : dynd::nd::base_callable(dynd::ndt::callable_type::make(bytes_id, {dynd::ndt::make_type<pyobject_type>()}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), dynd::nd::call_graph &cg,
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
      ckb->emplace_back<assign_from_pyobject_kernel<bytes_id>>(kernreq, dst_tp, dst_arrmeta);
    }
  };

  template <>
  class assign_from_pyobject_callable<fixed_bytes_id> : public dynd::nd::base_callable {
  public:
    assign_from_pyobject_callable()
        : dynd::nd::base_callable(
              dynd::ndt::callable_type::make(fixed_bytes_id, {dynd::ndt::make_type<pyobject_type>()}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), dynd::nd::call_graph &cg,
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
      ckb->emplace_back<assign_from_pyobject_kernel<fixed_bytes_id>>(kernreq, dst_tp, dst_arrmeta);
    }
  };

  template <>
  class assign_from_pyobject_callable<string_id> : public dynd::nd::base_callable {
  public:
    assign_from_pyobject_callable()
        : dynd::nd::base_callable(dynd::ndt::callable_type::make(string_id, {dynd::ndt::make_type<pyobject_type>()}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), dynd::nd::call_graph &cg,
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
      ckb->emplace_back<assign_from_pyobject_kernel<string_id>>(kernreq, dst_tp, dst_arrmeta);
    }
  };

  template <>
  class assign_from_pyobject_callable<fixed_string_id> : public dynd::nd::base_callable {
  public:
    assign_from_pyobject_callable()
        : dynd::nd::base_callable(
              dynd::ndt::callable_type::make(fixed_string_id, {dynd::ndt::make_type<pyobject_type>()}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), dynd::nd::call_graph &cg,
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
      ckb->emplace_back<assign_from_pyobject_kernel<fixed_string_id>>(kernreq, dst_tp, dst_arrmeta);
    }
  };

  template <>
  class assign_from_pyobject_callable<option_id> : public dynd::nd::base_callable {
  public:
    assign_from_pyobject_callable()
        : dynd::nd::base_callable(dynd::ndt::callable_type::make(option_id, {dynd::ndt::make_type<pyobject_type>()}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), dynd::nd::call_graph &cg,
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
      ckb->emplace_back<assign_from_pyobject_kernel<option_id>>(kernreq, dst_tp, dst_arrmeta);
      intptr_t ckb_offset = ckb->size();
      dynd::nd::assign_na->instantiate(NULL, ckb, dst_tp, dst_arrmeta, nsrc, NULL, NULL, dynd::kernel_request_single,
                                       nkwd, kwds, tp_vars);
      ckb_offset = ckb->size();
      ckb->get_at<assign_from_pyobject_kernel<option_id>>(root_ckb_offset)->copy_value_offset =
          ckb_offset - root_ckb_offset;
      dynd::nd::assign->instantiate(NULL, ckb, dst_tp.extended<dynd::ndt::option_type>()->get_value_type(), dst_arrmeta,
                                    nsrc, src_tp, src_arrmeta, dynd::kernel_request_single, nkwd, kwds, tp_vars);
      ckb_offset = ckb->size();
    }
  };

  template <>
  class assign_from_pyobject_callable<tuple_id> : public dynd::nd::base_callable {
  public:
    assign_from_pyobject_callable()
        : dynd::nd::base_callable(dynd::ndt::callable_type::make(tuple_id, {dynd::ndt::make_type<pyobject_type>()}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), dynd::nd::call_graph &cg,
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
      bool dim_broadcast = false;

      intptr_t ckb_offset = ckb->size();
      intptr_t root_ckb_offset = ckb_offset;
      ckb->emplace_back<assign_from_pyobject_kernel<tuple_id>>(kernreq);
      assign_from_pyobject_kernel<tuple_id> *self = ckb->get_at<assign_from_pyobject_kernel<tuple_id>>(root_ckb_offset);
      ckb_offset = ckb->size();
      self->m_dst_tp = dst_tp;
      self->m_dst_arrmeta = dst_arrmeta;
      intptr_t field_count = dst_tp.extended<dynd::ndt::tuple_type>()->get_field_count();
      const dynd::ndt::type *field_types = dst_tp.extended<dynd::ndt::tuple_type>()->get_field_types_raw();
      const uintptr_t *arrmeta_offsets = dst_tp.extended<dynd::ndt::tuple_type>()->get_arrmeta_offsets_raw();
      self->m_dim_broadcast = dim_broadcast;
      self->m_copy_el_offsets.resize(field_count);
      for (intptr_t i = 0; i < field_count; ++i) {
        ckb->reserve(ckb_offset);
        self = ckb->get_at<assign_from_pyobject_kernel<tuple_id>>(root_ckb_offset);
        self->m_copy_el_offsets[i] = ckb_offset - root_ckb_offset;
        const char *field_arrmeta = dst_arrmeta + arrmeta_offsets[i];
        dynd::nd::assign->instantiate(NULL, ckb, field_types[i], field_arrmeta, nsrc, src_tp, src_arrmeta,
                                      dynd::kernel_request_single, nkwd, kwds, tp_vars);
        ckb_offset = ckb->size();
      }
    }
  };

  template <>
  class assign_from_pyobject_callable<struct_id> : public dynd::nd::base_callable {
  public:
    assign_from_pyobject_callable()
        : dynd::nd::base_callable(dynd::ndt::callable_type::make(struct_id, {dynd::ndt::make_type<pyobject_type>()}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), dynd::nd::call_graph &cg,
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
      bool dim_broadcast = false;

      intptr_t ckb_offset = ckb->size();
      intptr_t root_ckb_offset = ckb_offset;
      ckb->emplace_back<assign_from_pyobject_kernel<struct_id>>(kernreq);
      assign_from_pyobject_kernel<struct_id> *self =
          ckb->get_at<assign_from_pyobject_kernel<struct_id>>(root_ckb_offset);
      ckb_offset = ckb->size();
      self->m_dst_tp = dst_tp;
      self->m_dst_arrmeta = dst_arrmeta;
      intptr_t field_count = dst_tp.extended<dynd::ndt::struct_type>()->get_field_count();
      const dynd::ndt::type *field_types = dst_tp.extended<dynd::ndt::struct_type>()->get_field_types_raw();
      const uintptr_t *arrmeta_offsets = dst_tp.extended<dynd::ndt::struct_type>()->get_arrmeta_offsets_raw();
      self->m_dim_broadcast = dim_broadcast;
      self->m_copy_el_offsets.resize(field_count);
      for (intptr_t i = 0; i < field_count; ++i) {
        ckb->reserve(ckb_offset);
        self = ckb->get_at<assign_from_pyobject_kernel<struct_id>>(root_ckb_offset);
        self->m_copy_el_offsets[i] = ckb_offset - root_ckb_offset;
        const char *field_arrmeta = dst_arrmeta + arrmeta_offsets[i];
        dynd::nd::assign->instantiate(NULL, ckb, field_types[i], field_arrmeta, nsrc, src_tp, src_arrmeta,
                                      dynd::kernel_request_single, nkwd, kwds, tp_vars);
        ckb_offset = ckb->size();
      }
    }
  };

  template <>
  class assign_from_pyobject_callable<fixed_dim_id> : public dynd::nd::base_callable {
  public:
    assign_from_pyobject_callable()
        : dynd::nd::base_callable(dynd::ndt::callable_type::make(fixed_dim_id, {dynd::ndt::make_type<pyobject_type>()}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), dynd::nd::call_graph &cg,
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
      if (dst_tp.get_as_strided(dst_arrmeta, &dim_size, &stride, &el_tp, &el_arrmeta)) {
        intptr_t root_ckb_offset = ckb_offset;
        ckb->emplace_back<assign_from_pyobject_kernel<fixed_dim_id>>(kernreq);
        assign_from_pyobject_kernel<fixed_dim_id> *self =
            ckb->get_at<assign_from_pyobject_kernel<fixed_dim_id>>(root_ckb_offset);
        ckb_offset = ckb->size();
        self->m_dim_size = dim_size;
        self->m_stride = stride;
        self->m_dst_tp = dst_tp;
        self->m_dst_arrmeta = dst_arrmeta;
        // from pyobject ckernel
        dynd::nd::assign->instantiate(NULL, ckb, el_tp, el_arrmeta, nsrc, src_tp, src_arrmeta,
                                      dynd::kernel_request_strided, nkwd, kwds, tp_vars);
        ckb_offset = ckb->size();
        self = ckb->get_at<assign_from_pyobject_kernel<fixed_dim_id>>(root_ckb_offset);
        self->m_copy_dst_offset = ckb_offset - root_ckb_offset;
        // dst to dst ckernel, for broadcasting case
        dynd::nd::array error_mode = assign_error_fractional;
        dynd::nd::assign->instantiate(NULL, ckb, el_tp, el_arrmeta, 1, &el_tp, &el_arrmeta,
                                      dynd::kernel_request_strided, 1, &error_mode, std::map<std::string, ndt::type>());

        return;
      }

      throw std::runtime_error("could not process as strided");
    }
  };

  template <>
  class assign_from_pyobject_callable<var_dim_id> : public dynd::nd::base_callable {
  public:
    assign_from_pyobject_callable()
        : dynd::nd::base_callable(dynd::ndt::callable_type::make(var_dim_id, {dynd::ndt::make_type<pyobject_type>()}))
    {
    }

    ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), dynd::nd::call_graph &cg,
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
      bool dim_broadcast = false;

      intptr_t root_ckb_offset = ckb_offset;
      ckb->emplace_back<assign_from_pyobject_kernel<var_dim_id>>(kernreq);
      assign_from_pyobject_kernel<var_dim_id> *self =
          ckb->get_at<assign_from_pyobject_kernel<var_dim_id>>(root_ckb_offset);
      ckb_offset = ckb->size();
      self->m_offset = reinterpret_cast<const dynd::ndt::var_dim_type::metadata_type *>(dst_arrmeta)->offset;
      self->m_stride = reinterpret_cast<const dynd::ndt::var_dim_type::metadata_type *>(dst_arrmeta)->stride;
      self->m_dst_tp = dst_tp;
      self->m_dst_arrmeta = dst_arrmeta;
      dynd::ndt::type el_tp = dst_tp.extended<dynd::ndt::var_dim_type>()->get_element_type();
      const char *el_arrmeta = dst_arrmeta + sizeof(dynd::ndt::var_dim_type::metadata_type);
      dynd::nd::assign->instantiate(NULL, ckb, el_tp, el_arrmeta, nsrc, src_tp, src_arrmeta,
                                    dynd::kernel_request_strided, nkwd, kwds, tp_vars);
      ckb_offset = ckb->size();
      self = ckb->get_at<assign_from_pyobject_kernel<var_dim_id>>(root_ckb_offset);
      self->m_copy_dst_offset = ckb_offset - root_ckb_offset;
      // dst to dst ckernel, for broadcasting case
      dynd::nd::array error_mode = assign_error_fractional;
      dynd::nd::assign->instantiate(NULL, ckb, el_tp, el_arrmeta, 1, &el_tp, &el_arrmeta, dynd::kernel_request_strided,
                                    1, &error_mode, std::map<std::string, ndt::type>());
    }
  };

} // namespace pydynd::nd
} // namespace pydynd

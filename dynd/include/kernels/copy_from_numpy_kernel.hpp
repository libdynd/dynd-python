#pragma once

#include <dynd/callables/base_callable.hpp>

using namespace dynd;

namespace pydynd {
namespace nd {

  class copy_from_numpy_callable : public dynd::nd::base_callable {
  public:
    copy_from_numpy_callable() : dynd::nd::base_callable(dynd::ndt::type("(void, broadcast: bool) -> T")) {}

    dynd::ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data),
                            dynd::nd::call_graph &cg, const dynd::ndt::type &dst_tp, size_t nsrc,
                            const dynd::ndt::type *src_tp, size_t nkwd, const dynd::nd::array *kwds,
                            const std::map<std::string, dynd::ndt::type> &tp_vars)
    {
      cg.emplace_back([=](dynd::nd::kernel_builder &kb, kernel_request_t kernreq, const char *dst_arrmeta, size_t nsrc,
                          const char *const *src_arrmeta) {
        if (src_tp[0].get_id() != dynd::void_id) {
          std::stringstream ss;
          ss << "Cannot instantiate dynd::nd::callable copy_from_numpy with "
                "signature (";
          ss << src_tp[0] << ") -> " << dst_tp;
          throw dynd::type_error(ss.str());
        }

        PyArray_Descr *dtype = *reinterpret_cast<PyArray_Descr *const *>(src_arrmeta[0]);
        uintptr_t src_alignment = reinterpret_cast<const uintptr_t *>(src_arrmeta[0])[1];

        if (!PyDataType_FLAGCHK(dtype, NPY_ITEM_HASOBJECT)) {
          // If there is no object type in the numpy type, get the dynd equivalent
          // type and use it to do the copying
          dynd::ndt::type src_view_tp = _type_from_numpy_dtype(dtype, src_alignment);
          dynd::nd::array error_mode = dynd::assign_error_fractional;
          //          dynd::nd::assign->instantiate(node, NULL, ckb, dst_tp, dst_arrmeta, 1, &src_view_tp, NULL,
          //          kernreq, 1,
          //                                      &error_mode, std::map<std::string, dynd::ndt::type>());

          return;
        }
        else if (PyDataType_ISOBJECT(dtype)) {
          dynd::nd::base_callable *af = dynd::nd::assign.get();
          dynd::ndt::type child_src_tp = dynd::ndt::make_type<pyobject_type>();
          //          af->instantiate(node, NULL, ckb, dst_tp, dst_arrmeta, 1, &child_src_tp, NULL, kernreq, nkwd, kwds,
          //          tp_vars);
          return;
        }
        else if (PyDataType_HASFIELDS(dtype)) {
          if (dst_tp.get_id() != dynd::struct_id && dst_tp.get_id() != dynd::tuple_id) {
            std::stringstream ss;
            ss << "Cannot assign from numpy type " << pyobject_repr((PyObject *)dtype) << " to dynd type " << dst_tp;
            throw std::invalid_argument(ss.str());
          }

          // Get the fields out of the numpy dtype
          std::vector<PyArray_Descr *> field_dtypes_orig;
          std::vector<std::string> field_names_orig;
          std::vector<size_t> field_offsets_orig;
          pydynd::extract_fields_from_numpy_struct(dtype, field_dtypes_orig, field_names_orig, field_offsets_orig);
          intptr_t field_count = field_dtypes_orig.size();
          if (field_count != dst_tp.extended<dynd::ndt::tuple_type>()->get_field_count()) {
            std::stringstream ss;
            ss << "Cannot assign from numpy type " << pyobject_repr((PyObject *)dtype) << " to dynd type " << dst_tp;
            throw std::invalid_argument(ss.str());
          }

          // Permute the numpy fields to match with the dynd fields
          std::vector<PyArray_Descr *> field_dtypes;
          std::vector<size_t> field_offsets;
          if (dst_tp.get_id() == dynd::struct_id) {
            field_dtypes.resize(field_count);
            field_offsets.resize(field_count);
            for (intptr_t i = 0; i < field_count; ++i) {
              intptr_t src_i = dst_tp.extended<dynd::ndt::struct_type>()->get_field_index(field_names_orig[i]);
              if (src_i >= 0) {
                field_dtypes[src_i] = field_dtypes_orig[i];
                field_offsets[src_i] = field_offsets_orig[i];
              }
              else {
                std::stringstream ss;
                ss << "Cannot assign from numpy type " << pyobject_repr((PyObject *)dtype) << " to dynd type "
                   << dst_tp;
                throw std::invalid_argument(ss.str());
              }
            }
          }
          else {
            // In the tuple case, use position instead of name
            field_dtypes.swap(field_dtypes_orig);
            field_offsets.swap(field_offsets_orig);
          }

          std::vector<dynd::ndt::type> src_fields_tp(field_count, dynd::ndt::make_type<void>());
          std::vector<copy_from_numpy_arrmeta> src_arrmeta_values(field_count);
          std::vector<const char *> src_fields_arrmeta(field_count);
          for (intptr_t i = 0; i < field_count; ++i) {
            src_arrmeta_values[i].src_dtype = field_dtypes[i];
            src_arrmeta_values[i].src_alignment = src_alignment | field_offsets[i];
            src_fields_arrmeta[i] = reinterpret_cast<const char *>(&src_arrmeta_values[i]);
          }

          const uintptr_t *dst_arrmeta_offsets = dst_tp.extended<dynd::ndt::tuple_type>()->get_arrmeta_offsets_raw();
          dynd::shortvector<const char *> dst_fields_arrmeta(field_count);
          for (intptr_t i = 0; i != field_count; ++i) {
            dst_fields_arrmeta[i] = dst_arrmeta + dst_arrmeta_offsets[i];
          }

          // Todo: Remove this line
          dynd::nd::callable af = dynd::nd::make_callable<copy_from_numpy_callable>();

          const std::vector<dynd::ndt::type> &dst_fields_tp =
              dst_tp.extended<dynd::ndt::tuple_type>()->get_field_types();
          const uintptr_t *dst_data_offsets = reinterpret_cast<const uintptr_t *>(dst_arrmeta);

          intptr_t self_offset = kb.size();
          kb.emplace_back<dynd::nd::tuple_unary_op_ck>(kernreq);
          dynd::nd::tuple_unary_op_ck *self = kb.get_at<dynd::nd::tuple_unary_op_ck>(self_offset);
          self->m_fields.resize(field_count);
          for (intptr_t i = 0; i < field_count; ++i) {
            self = kb.get_at<dynd::nd::tuple_unary_op_ck>(self_offset);
            dynd::nd::tuple_unary_op_item &field = self->m_fields[i];
            field.child_kernel_offset = kb.size() - self_offset;
            field.dst_data_offset = dst_data_offsets[i];
            field.src_data_offset = field_offsets[i];
            dynd::nd::array error_mode = dynd::ndt::traits<dynd::assign_error_mode>::na();
            //            af->instantiate(node, NULL, ckb, dst_fields_tp[i], dst_fields_arrmeta[i], 1,
            //            &src_fields_tp[i],
            //                          &src_fields_arrmeta[i], dynd::kernel_request_single, 1, &error_mode,
            //                        std::map<std::string, dynd::ndt::type>());
          }

          return;
        }
        else {
          std::stringstream ss;
          ss << "TODO: implement assign from numpy type " << pyobject_repr((PyObject *)dtype) << " to dynd type "
             << dst_tp;
          throw std::invalid_argument(ss.str());
        }
      });

      return dst_tp;
    }
  };

} // namespace pydynd::nd
} // namespace pydynd

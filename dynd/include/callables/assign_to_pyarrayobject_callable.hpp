#pragma once

#include "kernels/assign_to_pyarrayobject_kernel.hpp"
#include <dynd/callables/base_callable.hpp>

/**
 * This sets up a ckernel to copy from a dynd array
 * to a numpy array. The destination numpy array is
 * represented by dst_tp being ``void`` and the dst_arrmeta
 * being a pointer to the ``PyArray_Descr *`` of the type for the destination.
 */
class assign_to_pyarrayobject_callable : public dynd::nd::base_callable {
public:
  assign_to_pyarrayobject_callable() : base_callable(dynd::ndt::type("(Any) -> void")) {}

  ndt::type resolve(dynd::nd::base_callable *DYND_UNUSED(caller), char *DYND_UNUSED(data),
                    dynd::nd::call_graph &DYND_UNUSED(cg), const dynd::ndt::type &dst_tp, size_t DYND_UNUSED(nsrc),
                    const dynd::ndt::type *DYND_UNUSED(src_tp), size_t DYND_UNUSED(nkwd),
                    const dynd::nd::array *DYND_UNUSED(kwds),
                    const std::map<std::string, dynd::ndt::type> &DYND_UNUSED(tp_vars))
  {
    return dst_tp;
  }

  /*
    void instantiate(char *data, dynd::nd::kernel_builder *ckb, const dynd::ndt::type &dst_tp, const char *dst_arrmeta,
                     intptr_t nsrc, const dynd::ndt::type *src_tp, const char *const *src_arrmeta,
                     dynd::kernel_request_t kernreq, intptr_t nkwd, const dynd::nd::array *kwds,
                     const std::map<std::string, dynd::ndt::type> &tp_vars)
    {
      PyObject *dst_obj = *reinterpret_cast<PyObject *const *>(dst_arrmeta);
      uintptr_t dst_alignment = reinterpret_cast<const uintptr_t *>(dst_arrmeta)[1];

      PyArray_Descr *dtype = reinterpret_cast<PyArray_Descr *>(dst_obj);

      // If there is no object type in the numpy type, get the dynd equivalent
      // type and use it to do the copying
      if (!PyDataType_FLAGCHK(dtype, NPY_ITEM_HASOBJECT)) {
        dynd::ndt::type dst_view_tp = pydynd::_type_from_numpy_dtype(dtype, dst_alignment);
        nd::array error_mode = assign_error_fractional;
        nd::assign->instantiate(node, NULL, ckb, dst_view_tp, NULL, 1, src_tp, src_arrmeta, kernreq, 1, &error_mode,
                                std::map<std::string, ndt::type>());
        return;
      }

      if (PyDataType_ISOBJECT(dtype)) {
        dynd::nd::assign->instantiate(node, NULL, ckb, dynd::ndt::make_type<pyobject_type>(), NULL, nsrc, src_tp,
                                      src_arrmeta, kernreq, 0, NULL, tp_vars);
        return;
      }

      if (PyDataType_HASFIELDS(dtype)) {
        if (src_tp[0].get_id() != dynd::struct_id && src_tp[0].get_id() != dynd::tuple_id) {
          std::stringstream ss;
          pydynd::py_ref dtype_str = capture_if_not_null(PyObject_Str((PyObject *)dtype));
          ss << "Cannot assign from source dynd type " << src_tp[0] << " to numpy type "
             << pydynd::pystring_as_string(dtype_str.get());
          throw std::invalid_argument(ss.str());
        }

        // Get the fields out of the numpy dtype
        std::vector<PyArray_Descr *> field_dtypes_orig;
        std::vector<std::string> field_names_orig;
        std::vector<size_t> field_offsets_orig;
        pydynd::extract_fields_from_numpy_struct(dtype, field_dtypes_orig, field_names_orig, field_offsets_orig);
        intptr_t field_count = field_dtypes_orig.size();
        if (field_count != src_tp[0].extended<dynd::ndt::tuple_type>()->get_field_count()) {
          std::stringstream ss;
          pydynd::py_ref dtype_str = capture_if_not_null(PyObject_Str((PyObject *)dtype));
          ss << "Cannot assign from source dynd type " << src_tp[0] << " to numpy type "
             << pydynd::pystring_as_string(dtype_str.get());
          throw std::invalid_argument(ss.str());
        }

        // Permute the numpy fields to match with the dynd fields
        std::vector<PyArray_Descr *> field_dtypes;
        std::vector<size_t> field_offsets;
        if (src_tp[0].get_id() == dynd::struct_id) {
          field_dtypes.resize(field_count);
          field_offsets.resize(field_count);
          for (intptr_t i = 0; i < field_count; ++i) {
            intptr_t src_i = src_tp[0].extended<dynd::ndt::struct_type>()->get_field_index(field_names_orig[i]);
            if (src_i >= 0) {
              field_dtypes[src_i] = field_dtypes_orig[i];
              field_offsets[src_i] = field_offsets_orig[i];
            }
            else {
              std::stringstream ss;
              pydynd::py_ref dtype_str = capture_if_not_null(PyObject_Str((PyObject *)dtype));
              ss << "Cannot assign from source dynd type " << src_tp[0] << " to numpy type "
                 << pydynd::pystring_as_string(dtype_str.get());
              throw std::invalid_argument(ss.str());
            }
          }
        }
        else {
          // In the tuple case, use position instead of name
          field_dtypes.swap(field_dtypes_orig);
          field_offsets.swap(field_offsets_orig);
        }

        std::vector<dynd::ndt::type> dst_fields_tp(field_count, dynd::ndt::make_type<void>());
        std::vector<copy_to_numpy_arrmeta> dst_arrmeta_values(field_count);
        std::vector<const char *> dst_fields_arrmeta(field_count);
        for (intptr_t i = 0; i < field_count; ++i) {
          dst_arrmeta_values[i].dst_dtype = field_dtypes[i];
          dst_arrmeta_values[i].dst_alignment = dst_alignment | field_offsets[i];
          dst_fields_arrmeta[i] = reinterpret_cast<const char *>(&dst_arrmeta_values[i]);
        }

        const uintptr_t *src_arrmeta_offsets = src_tp[0].extended<dynd::ndt::tuple_type>()->get_arrmeta_offsets_raw();
        dynd::shortvector<const char *> src_fields_arrmeta(field_count);
        for (intptr_t i = 0; i != field_count; ++i) {
          src_fields_arrmeta[i] = src_arrmeta[0] + src_arrmeta_offsets[i];
        }

        // Todo: Remove this
        dynd::nd::callable af = dynd::nd::make_callable<assign_to_pyarrayobject_callable>();

        const std::vector<ndt::type> &src_field_tp = src_tp[0].extended<dynd::ndt::tuple_type>()->get_field_types();
        const uintptr_t *src_data_offsets =
    src_tp[0].extended<dynd::ndt::tuple_type>()->get_data_offsets(src_arrmeta[0]);

        intptr_t self_offset = ckb->size();
        ckb->emplace_back<nd::tuple_unary_op_ck>(kernreq);
        nd::tuple_unary_op_ck *self = ckb->get_at<nd::tuple_unary_op_ck>(self_offset);
        self->m_fields.resize(field_count);
        for (intptr_t i = 0; i < field_count; ++i) {
          self = ckb->get_at<nd::tuple_unary_op_ck>(self_offset);
          nd::tuple_unary_op_item &field = self->m_fields[i];
          field.child_kernel_offset = ckb->size() - self_offset;
          field.dst_data_offset = field_offsets[i];
          field.src_data_offset = src_data_offsets[i];
          nd::array error_mode = ndt::traits<assign_error_mode>::na();
          af->instantiate(node, NULL, ckb, dst_fields_tp[i], dst_fields_arrmeta[i], 1, &src_field_tp[i],
                          &src_fields_arrmeta[i], kernel_request_single, 1, &error_mode,
                          std::map<std::string, ndt::type>());
        }
        return;
      }
      else {
        std::stringstream ss;
        ss << "TODO: implement assign from source dynd type " << src_tp[0] << " to numpy type "
           << pydynd::pyobject_repr((PyObject *)dtype);
        throw std::invalid_argument(ss.str());
      }
    }
  */
};

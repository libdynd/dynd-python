#pragma once

#include "config.hpp"
#include <dynd/kernels/expr_kernels.hpp>

namespace pydynd {
namespace nd {
  namespace functional {

    struct scalar_ufunc_data {
      PyUFuncObject *ufunc;
      PyUFuncGenericFunction funcptr;
      void *ufunc_data;
      intptr_t param_count;

      ~scalar_ufunc_data()
      {
        if (ufunc != NULL) {
          // Acquire the GIL for the python decref
          PyGILState_RAII pgs;
          Py_DECREF(ufunc);
        }
      }
    };

    template <bool gil>
    struct scalar_ufunc_ck;

    template <>
    struct scalar_ufunc_ck<false>
        : expr_ck<scalar_ufunc_ck<false>, kernel_request_host, 1> {
      typedef scalar_ufunc_ck self_type;

      const scalar_ufunc_data *data;

      scalar_ufunc_ck(const scalar_ufunc_data *data) : data(data) {}

      void single(char *dst, char *const *src)
      {
        char *args[NPY_MAXARGS];
        // Set up the args array the way the numpy ufunc wants it
        memcpy(&args[0], &src[0], data->param_count * sizeof(void *));
        args[data->param_count] = dst;
        // Call the ufunc loop with a dim size of 1
        intptr_t dimsize = 1;
        intptr_t strides[NPY_MAXARGS];
        memset(strides, 0, (data->param_count + 1) * sizeof(intptr_t));
        data->funcptr(args, &dimsize, strides, data->ufunc_data);
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src,
                   const intptr_t *src_stride, size_t count)
      {
        char *args[NPY_MAXARGS];
        // Set up the args array the way the numpy ufunc wants it
        memcpy(&args[0], &src[0], data->param_count * sizeof(void *));
        args[data->param_count] = dst;
        // Call the ufunc loop with a dim size of 1
        intptr_t strides[NPY_MAXARGS];
        memcpy(&strides[0], &src_stride[0],
               data->param_count * sizeof(intptr_t));
        strides[data->param_count] = dst_stride;
        data->funcptr(args, reinterpret_cast<intptr_t *>(&count), strides,
                      data->ufunc_data);
      }

      static intptr_t instantiate(
          const arrfunc_type_data *af_self, const arrfunc_type *af_tp,
          char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
          const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta),
          intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
          const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
          const eval::eval_context *DYND_UNUSED(ectx), const nd::array &kwds,
          const std::map<nd::string, ndt::type> &tp_vars)
      {
        if (dst_tp != af_tp->get_return_type()) {
          std::stringstream ss;
          ss << "destination type requested, " << dst_tp
             << ", does not match the ufunc's type "
             << af_tp->get_return_type();
          throw type_error(ss.str());
        }
        intptr_t param_count = af_tp->get_npos();
        for (intptr_t i = 0; i != param_count; ++i) {
          if (src_tp[i] != af_tp->get_pos_type(i)) {
            std::stringstream ss;
            ss << "source type requested for parameter " << (i + 1) << ", "
               << src_tp[i] << ", does not match the ufunc's type "
               << af_tp->get_pos_type(i);
            throw type_error(ss.str());
          }
        }

        // Acquire the GIL for creating the ckernel
        PyGILState_RAII pgs;
        self_type::create(ckb, kernreq, ckb_offset,
                          *af_self->get_data_as<scalar_ufunc_data *>());
        return ckb_offset;
      }

      static void free(arrfunc_type_data *self_af)
      {
        scalar_ufunc_data *data = *self_af->get_data_as<scalar_ufunc_data *>();
        // Call the destructor and free the memory
        delete data;
      }
    };

    template <>
    struct scalar_ufunc_ck<true>
        : expr_ck<scalar_ufunc_ck<true>, kernel_request_host, 1> {
      typedef scalar_ufunc_ck self_type;

      const scalar_ufunc_data *data;

      scalar_ufunc_ck(const scalar_ufunc_data *data) : data(data) {}

      void single(char *dst, char *const *src)
      {
        char *args[NPY_MAXARGS];
        // Set up the args array the way the numpy ufunc wants it
        memcpy(&args[0], &src[0], data->param_count * sizeof(void *));
        args[data->param_count] = dst;
        // Call the ufunc loop with a dim size of 1
        intptr_t dimsize = 1;
        intptr_t strides[NPY_MAXARGS];
        memset(strides, 0, (data->param_count + 1) * sizeof(void *));
        {
          PyGILState_RAII pgs;
          data->funcptr(args, &dimsize, strides, data->ufunc_data);
        }
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src,
                   const intptr_t *src_stride, size_t count)
      {
        char *args[NPY_MAXARGS];
        // Set up the args array the way the numpy ufunc wants it
        memcpy(&args[0], &src[0], data->param_count * sizeof(void *));
        args[data->param_count] = dst;
        // Call the ufunc loop with a dim size of 1
        intptr_t strides[NPY_MAXARGS];
        memcpy(&strides[0], &src_stride[0],
               data->param_count * sizeof(intptr_t));
        strides[data->param_count] = dst_stride;
        {
          PyGILState_RAII pgs;
          data->funcptr(args, reinterpret_cast<intptr_t *>(&count), strides,
                        data->ufunc_data);
        }
      }

      static intptr_t instantiate(
          const arrfunc_type_data *af_self, const arrfunc_type *af_tp,
          char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
          const ndt::type &dst_tp, const char *DYND_UNUSED(dst_arrmeta),
          intptr_t DYND_UNUSED(nsrc), const ndt::type *src_tp,
          const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
          const eval::eval_context *DYND_UNUSED(ectx), const nd::array &kwds,
          const std::map<nd::string, ndt::type> &tp_vars)
      {
        if (dst_tp != af_tp->get_return_type()) {
          std::stringstream ss;
          ss << "destination type requested, " << dst_tp
             << ", does not match the ufunc's type "
             << af_tp->get_return_type();
          throw type_error(ss.str());
        }
        intptr_t param_count = af_tp->get_npos();
        for (intptr_t i = 0; i != param_count; ++i) {
          if (src_tp[i] != af_tp->get_pos_type(i)) {
            std::stringstream ss;
            ss << "source type requested for parameter " << (i + 1) << ", "
               << src_tp[i] << ", does not match the ufunc's type "
               << af_tp->get_pos_type(i);
            throw type_error(ss.str());
          }
        }

        // Acquire the GIL for creating the ckernel
        PyGILState_RAII pgs;
        self_type::create(ckb, kernreq, ckb_offset,
                          *af_self->get_data_as<scalar_ufunc_data *>());
        return ckb_offset;
      }

      static void free(arrfunc_type_data *self_af)
      {
        scalar_ufunc_data *data = *self_af->get_data_as<scalar_ufunc_data *>();
        // Call the destructor and free the memory
        delete data;
      }
    };

  } // namespace pydynd::nd::functional
} // namespace pydynd::nd
} // namespace pydynd
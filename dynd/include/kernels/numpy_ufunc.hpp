#pragma once

#include "config.hpp"
#include <dynd/kernels/base_kernel.hpp>

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
    struct scalar_ufunc_ck<false> : dynd::nd::base_kernel<
                                        scalar_ufunc_ck<false>, 1> {
      typedef scalar_ufunc_ck self_type;

      const scalar_ufunc_data *data;

      scalar_ufunc_ck(const scalar_ufunc_data *data) : data(data)
      {
      }

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

      static intptr_t
      instantiate(char *static_data, size_t DYND_UNUSED(data_size),
                  char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                  const dynd::ndt::type &dst_tp,
                  const char *DYND_UNUSED(dst_arrmeta),
                  intptr_t DYND_UNUSED(nsrc), const dynd::ndt::type *src_tp,
                  const char *const *DYND_UNUSED(src_arrmeta),
                  dynd::kernel_request_t kernreq,
                  const dynd::eval::eval_context *DYND_UNUSED(ectx),
                  intptr_t nkwd, const dynd::nd::array *kwds,
                  const std::map<std::string, dynd::ndt::type> &tp_vars)
      {
        // Acquire the GIL for creating the ckernel
        PyGILState_RAII pgs;
        self_type::make(ckb, kernreq, ckb_offset,
                        reinterpret_cast<std::shared_ptr<scalar_ufunc_data> *>(
                            static_data)->get());
        return ckb_offset;
      }
    };

    template <>
    struct scalar_ufunc_ck<true> : dynd::nd::base_kernel<scalar_ufunc_ck<true>,
                                                         1> {
      typedef scalar_ufunc_ck self_type;

      const scalar_ufunc_data *data;

      scalar_ufunc_ck(const scalar_ufunc_data *data) : data(data)
      {
      }

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

      static intptr_t
      instantiate(char *static_data, size_t DYND_UNUSED(data_size),
                  char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                  const dynd::ndt::type &dst_tp,
                  const char *DYND_UNUSED(dst_arrmeta),
                  intptr_t DYND_UNUSED(nsrc), const dynd::ndt::type *src_tp,
                  const char *const *DYND_UNUSED(src_arrmeta),
                  dynd::kernel_request_t kernreq,
                  const dynd::eval::eval_context *DYND_UNUSED(ectx),
                  intptr_t nkwd, const dynd::nd::array *kwds,
                  const std::map<std::string, dynd::ndt::type> &tp_vars)
      {
        // Acquire the GIL for creating the ckernel
        PyGILState_RAII pgs;
        self_type::make(ckb, kernreq, ckb_offset,
                        reinterpret_cast<std::shared_ptr<scalar_ufunc_data> *>(
                            static_data)->get());
        return ckb_offset;
      }
    };

  } // namespace pydynd::nd::functional
} // namespace pydynd::nd
} // namespace pydynd

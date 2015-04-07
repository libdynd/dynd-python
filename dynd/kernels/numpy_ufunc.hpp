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
      int ckernel_acquires_gil;
      intptr_t param_count;
    };

    template <bool gil>
    struct scalar_ufunc_ck;

    template <>
    struct scalar_ufunc_ck<true>
        : expr_ck<scalar_ufunc_ck<true>, kernel_request_host, 1> {
      PyUFuncGenericFunction funcptr;
      void *ufunc_data;
      intptr_t param_count;
      PyUFuncObject *ufunc;

      scalar_ufunc_ck(PyUFuncGenericFunction funcptr, void *ufunc_data,
                      intptr_t param_count, PyUFuncObject *ufunc)
          : funcptr(funcptr), ufunc_data(ufunc_data), param_count(param_count),
            ufunc(ufunc)
      {
      }

      ~scalar_ufunc_ck()
      {
        if (ufunc != NULL) {
          // Acquire the GIL for the python decref
          PyGILState_RAII pgs;
          Py_DECREF(ufunc);
        }
      }

      void single(char *dst, char *const *src)
      {
        char *args[NPY_MAXARGS];
        // Set up the args array the way the numpy ufunc wants it
        memcpy(&args[0], &src[0], param_count * sizeof(void *));
        args[param_count] = dst;
        // Call the ufunc loop with a dim size of 1
        intptr_t dimsize = 1;
        intptr_t strides[NPY_MAXARGS];
        memset(strides, 0, (param_count + 1) * sizeof(void *));
        {
          PyGILState_RAII pgs;
          funcptr(args, &dimsize, strides, ufunc_data);
        }
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src,
                   const intptr_t *src_stride, size_t count)
      {
        char *args[NPY_MAXARGS];
        // Set up the args array the way the numpy ufunc wants it
        memcpy(&args[0], &src[0], param_count * sizeof(void *));
        args[param_count] = dst;
        // Call the ufunc loop with a dim size of 1
        intptr_t strides[NPY_MAXARGS];
        memcpy(&strides[0], &src_stride[0], param_count * sizeof(intptr_t));
        strides[param_count] = dst_stride;
        {
          PyGILState_RAII pgs;
          funcptr(args, reinterpret_cast<intptr_t *>(&count), strides,
                  ufunc_data);
        }
      }
    };

    template <>
    struct scalar_ufunc_ck<false>
        : expr_ck<scalar_ufunc_ck<false>, kernel_request_host, 1> {
      PyUFuncGenericFunction funcptr;
      void *ufunc_data;
      intptr_t param_count;
      PyUFuncObject *ufunc;

      scalar_ufunc_ck(PyUFuncGenericFunction funcptr, void *ufunc_data,
                      intptr_t param_count, PyUFuncObject *ufunc)
          : funcptr(funcptr), ufunc_data(ufunc_data), param_count(param_count),
            ufunc(ufunc)
      {
      }

      ~scalar_ufunc_ck()
      {
        if (ufunc != NULL) {
          // Acquire the GIL for the python decref
          PyGILState_RAII pgs;
          Py_DECREF(ufunc);
        }
      }

      void single(char *dst, char *const *src)
      {
        char *args[NPY_MAXARGS];
        // Set up the args array the way the numpy ufunc wants it
        memcpy(&args[0], &src[0], param_count * sizeof(void *));
        args[param_count] = dst;
        // Call the ufunc loop with a dim size of 1
        intptr_t dimsize = 1;
        intptr_t strides[NPY_MAXARGS];
        memset(strides, 0, (param_count + 1) * sizeof(intptr_t));
        funcptr(args, &dimsize, strides, ufunc_data);
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src,
                   const intptr_t *src_stride, size_t count)
      {
        char *args[NPY_MAXARGS];
        // Set up the args array the way the numpy ufunc wants it
        memcpy(&args[0], &src[0], param_count * sizeof(void *));
        args[param_count] = dst;
        // Call the ufunc loop with a dim size of 1
        intptr_t strides[NPY_MAXARGS];
        memcpy(&strides[0], &src_stride[0], param_count * sizeof(intptr_t));
        strides[param_count] = dst_stride;
        funcptr(args, reinterpret_cast<intptr_t *>(&count), strides,
                ufunc_data);
      }
    };

  } // namespace pydynd::nd::functional
} // namespace pydynd::nd
} // namespace pydynd
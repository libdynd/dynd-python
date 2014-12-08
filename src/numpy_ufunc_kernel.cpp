//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//


#include "numpy_ufunc_kernel.hpp"

#if defined(_MSC_VER)
#pragma warning(push,2)
#endif
#include <numpy/npy_3kcompat.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include <dynd/kernels/expr_kernels.hpp>
#include <dynd/types/arrfunc_old_type.hpp>

#include "exception_translation.hpp"
#include "utility_functions.hpp"
#include "array_functions.hpp"

#if DYND_NUMPY_INTEROP

using namespace std;
using namespace dynd;
using namespace pydynd;

PyObject *pydynd::numpy_typetuples_from_ufunc(PyObject *ufunc)
{
    // NOTE: This function does not raise C++ exceptions,
    //       it behaves as a Python C-API function.
    if (!PyObject_TypeCheck(ufunc, &PyUFunc_Type)) {
        stringstream ss;
        ss << "a numpy ufunc object is required to retrieve type tuples, ";
        pyobject_ownref repr_obj(PyObject_Repr(ufunc));
        ss << "got " << pystring_as_string(repr_obj.get());
        PyErr_SetString(PyExc_TypeError, ss.str().c_str());
        return NULL;
    }
    PyUFuncObject *uf = (PyUFuncObject *)ufunc;

    // Process the main ufunc loops list
    int builtin_count = uf->ntypes;
    int nargs = uf->nin + uf->nout;
    PyObject *result = PyList_New(builtin_count);
    if (result == NULL) {
        return NULL;
    }
    for (int i = 0; i < builtin_count; ++i) {
        PyObject *typetup = PyTuple_New(nargs);
        if (typetup == NULL) {
            Py_DECREF(result);
            return NULL;
        }
        char *types = uf->types + i * nargs;
        // Switch from numpy convention "in, out" to dynd convention "out, in"
        {
            PyObject *descr = (PyObject *)PyArray_DescrFromType(types[nargs-1]);
            if (descr == NULL) {
                Py_DECREF(result);
                Py_DECREF(typetup);
                return NULL;
            }
            PyTuple_SET_ITEM(typetup, 0, descr);
        }
        for (int j = 1; j < nargs; ++j) {
            PyObject *descr = (PyObject *)PyArray_DescrFromType(types[j-1]);
            if (descr == NULL) {
                Py_DECREF(result);
                Py_DECREF(typetup);
                return NULL;
            }
            PyTuple_SET_ITEM(typetup, j, descr);
        }
        PyList_SET_ITEM(result, i, typetup);
    }

    // Process the userloops as well
    if (uf->userloops != NULL) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(uf->userloops, &pos, &key, &value)) {
            PyUFunc_Loop1d *funcdata = (PyUFunc_Loop1d *)NpyCapsule_AsVoidPtr(value);
            while (funcdata != NULL) {
                PyObject *typetup = PyTuple_New(nargs);
                if (typetup == NULL) {
                    Py_DECREF(result);
                    return NULL;
                }
                int *types = funcdata->arg_types;
                // Switch from numpy convention "in, out" to dynd convention "out, in"
                {
                    PyObject *descr = (PyObject *)PyArray_DescrFromType(types[nargs-1]);
                    if (descr == NULL) {
                        Py_DECREF(result);
                        Py_DECREF(typetup);
                        return NULL;
                    }
                    PyTuple_SET_ITEM(typetup, 0, descr);
                }
                for (int j = 1; j < nargs; ++j) {
                    PyObject *descr = (PyObject *)PyArray_DescrFromType(types[j-1]);
                    if (descr == NULL) {
                        Py_DECREF(result);
                        Py_DECREF(typetup);
                        return NULL;
                    }
                    PyTuple_SET_ITEM(typetup, j, descr);
                }
                PyList_Append(result, typetup);

                funcdata = funcdata->next;
            }
        }
    }

    return result;
}

namespace {
struct scalar_ufunc_data {
  PyUFuncObject *ufunc;
  PyUFuncGenericFunction funcptr;
  void *ufunc_data;
  int ckernel_acquires_gil;
  intptr_t param_count;
};

static void delete_scalar_ufunc_data(arrfunc_type_data *self_af)
{
  scalar_ufunc_data *data = *self_af->get_data_as<scalar_ufunc_data *>();
  if (data->ufunc != NULL) {
    // Acquire the GIL for the python decref
    PyGILState_RAII pgs;
    Py_DECREF(data->ufunc);
  }
  // Call the destructor and free the memory
  delete data;
}

struct scalar_ufunc_ckernel_data {
  ckernel_prefix base;
  PyUFuncGenericFunction funcptr;
  void *ufunc_data;
  intptr_t param_count;
  PyUFuncObject *ufunc;
};

static void delete_scalar_ufunc_ckernel_data(ckernel_prefix *self_data_ptr)
{
  scalar_ufunc_ckernel_data *data =
      reinterpret_cast<scalar_ufunc_ckernel_data *>(self_data_ptr);
  if (data->ufunc != NULL) {
    // Acquire the GIL for the python decref
    PyGILState_RAII pgs;
    Py_DECREF(data->ufunc);
  }
}

static void scalar_ufunc_single_ckernel_acquiregil(char *dst, char *const *src,
                                                   ckernel_prefix *ckp)
{
  scalar_ufunc_ckernel_data *data =
      reinterpret_cast<scalar_ufunc_ckernel_data *>(ckp);
  char *args[NPY_MAXARGS];
  intptr_t param_count = data->param_count;
  // Set up the args array the way the numpy ufunc wants it
  memcpy(&args[0], &src[0], param_count * sizeof(void *));
  args[param_count] = dst;
  // Call the ufunc loop with a dim size of 1
  intptr_t dimsize = 1;
  intptr_t strides[NPY_MAXARGS];
  memset(strides, 0, (param_count + 1) * sizeof(void *));
  {
    PyGILState_RAII pgs;
    data->funcptr(args, &dimsize, strides, data->ufunc_data);
  }
}

static void scalar_ufunc_single_ckernel_nogil(char *dst, char *const *src,
                                              ckernel_prefix *ckp)
{
  scalar_ufunc_ckernel_data *data =
      reinterpret_cast<scalar_ufunc_ckernel_data *>(ckp);
  char *args[NPY_MAXARGS];
  intptr_t param_count = data->param_count;
  // Set up the args array the way the numpy ufunc wants it
  memcpy(&args[0], &src[0], param_count * sizeof(void *));
  args[param_count] = dst;
  // Call the ufunc loop with a dim size of 1
  intptr_t dimsize = 1;
  intptr_t strides[NPY_MAXARGS];
  memset(strides, 0, (param_count + 1) * sizeof(intptr_t));
  data->funcptr(args, &dimsize, strides, data->ufunc_data);
}

static void
scalar_ufunc_strided_ckernel_acquiregil(char *dst, intptr_t dst_stride,
                                        char *const *src, const intptr_t *src_stride,
                                        size_t count, ckernel_prefix *ckp)
{
  scalar_ufunc_ckernel_data *data =
      reinterpret_cast<scalar_ufunc_ckernel_data *>(ckp);
  char *args[NPY_MAXARGS];
  intptr_t param_count = data->param_count;
  // Set up the args array the way the numpy ufunc wants it
  memcpy(&args[0], &src[0], param_count * sizeof(void *));
  args[param_count] = dst;
  // Call the ufunc loop with a dim size of 1
  intptr_t strides[NPY_MAXARGS];
  memcpy(&strides[0], &src_stride[0], param_count * sizeof(intptr_t));
  strides[param_count] = dst_stride;
  {
    PyGILState_RAII pgs;
    data->funcptr(args, reinterpret_cast<intptr_t *>(&count), strides,
                  data->ufunc_data);
  }
}

static void scalar_ufunc_strided_ckernel_nogil(char *dst, intptr_t dst_stride,
                                               char *const *src,
                                               const intptr_t *src_stride,
                                               size_t count,
                                               ckernel_prefix *ckp)
{
  scalar_ufunc_ckernel_data *data =
      reinterpret_cast<scalar_ufunc_ckernel_data *>(ckp);
  char *args[NPY_MAXARGS];
  intptr_t param_count = data->param_count;
  // Set up the args array the way the numpy ufunc wants it
  memcpy(&args[0], &src[0], param_count * sizeof(void *));
  args[param_count] = dst;
  // Call the ufunc loop with a dim size of 1
  intptr_t strides[NPY_MAXARGS];
  memcpy(&strides[0], &src_stride[0], param_count * sizeof(intptr_t));
  strides[param_count] = dst_stride;
  data->funcptr(args, reinterpret_cast<intptr_t *>(&count), strides,
                data->ufunc_data);
}

static intptr_t instantiate_scalar_ufunc_ckernel(
    const arrfunc_type_data *af_self, const arrfunc_type *af_tp,
    void *ckb, intptr_t ckb_offset, const ndt::type &dst_tp,
    const char *DYND_UNUSED(dst_arrmeta), const ndt::type *src_tp,
    const char *const *DYND_UNUSED(src_arrmeta), kernel_request_t kernreq,
    const eval::eval_context *DYND_UNUSED(ectx), const nd::array &kwds)
{
  if (dst_tp != af_tp->get_return_type()) {
    stringstream ss;
    ss << "destination type requested, " << dst_tp
       << ", does not match the ufunc's type " << af_tp->get_return_type();
    throw type_error(ss.str());
  }
  intptr_t param_count = af_tp->get_nsrc();
  for (intptr_t i = 0; i != param_count; ++i) {
    if (src_tp[i] != af_tp->get_arg_type(i)) {
      stringstream ss;
      ss << "source type requested for parameter " << (i + 1) << ", "
         << src_tp[i] << ", does not match the ufunc's type "
         << af_tp->get_arg_type(i);
      throw type_error(ss.str());
    }
  }

  if (!kwds.is_null()) {
    throw invalid_argument("unexpected non-NULL kwds value to "
                           "numpy ufunc/arrfunc adapter instantiation");
  }

  // Acquire the GIL for creating the ckernel
  PyGILState_RAII pgs;
  scalar_ufunc_data *data = *af_self->get_data_as<scalar_ufunc_data *>();
  scalar_ufunc_ckernel_data *af =
      reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb)->alloc_ck_leaf<scalar_ufunc_ckernel_data>(ckb_offset);
  af->base.destructor = &delete_scalar_ufunc_ckernel_data;
  if (data->ckernel_acquires_gil) {
    af->base.set_expr_function(kernreq, &scalar_ufunc_single_ckernel_acquiregil,
                               &scalar_ufunc_strided_ckernel_acquiregil);
  }
  else {
    af->base.set_expr_function(kernreq, &scalar_ufunc_single_ckernel_nogil,
                               &scalar_ufunc_strided_ckernel_nogil);
  }
  af->funcptr = data->funcptr;
  af->ufunc_data = data->ufunc_data;
  af->param_count = data->param_count;
  af->ufunc = data->ufunc;
  Py_INCREF(af->ufunc);
  return ckb_offset;
}

} // anonymous namespace

PyObject *pydynd::arrfunc_from_ufunc(PyObject *ufunc, PyObject *type_tuple,
                                     int ckernel_acquires_gil)
{
  try {
    // NOTE: This function does not raise C++ exceptions,
    //       it behaves as a Python C-API function.
    if (!PyObject_TypeCheck(ufunc, &PyUFunc_Type)) {
      stringstream ss;
      ss << "a numpy ufunc object is required by this function to create a "
            "arrfunc, ";
      pyobject_ownref repr_obj(PyObject_Repr(ufunc));
      ss << "got " << pystring_as_string(repr_obj.get());
      PyErr_SetString(PyExc_TypeError, ss.str().c_str());
      return NULL;
    }
    PyUFuncObject *uf = (PyUFuncObject *)ufunc;
    if (uf->nout != 1) {
      PyErr_SetString(PyExc_TypeError, "numpy ufuncs with multiple return "
                                       "arguments are not supported");
      return NULL;
    }
    if (uf->data == (void *)PyUFunc_SetUsesArraysAsData) {
      PyErr_SetString(PyExc_TypeError, "numpy ufuncs which require "
                                       "arrays as their data is not supported");
      return NULL;
    }

    // Convert the type tuple to an integer type_num array
    if (!PyTuple_Check(type_tuple)) {
      PyErr_SetString(PyExc_TypeError, "type_tuple must be a tuple");
      return NULL;
    }
    intptr_t nargs = PyTuple_GET_SIZE(type_tuple);
    if (nargs != uf->nin + uf->nout) {
      PyErr_SetString(PyExc_ValueError,
                      "type_tuple has the wrong size for the ufunc");
      return NULL;
    }
    int argtypes[NPY_MAXARGS];
    for (intptr_t i = 0; i < nargs; ++i) {
      PyArray_Descr *dt = NULL;
      if (!PyArray_DescrConverter(PyTuple_GET_ITEM(type_tuple, i), &dt)) {
        return NULL;
      }
      argtypes[i] = dt->type_num;
      Py_DECREF(dt);
    }

    // Search through the main loops for one that matches
    int builtin_count = uf->ntypes;
    for (int i = 0; i < builtin_count; ++i) {
      char *types = uf->types + i * nargs;
      bool matched = true;
      // Match the numpy convention "in, out" vs the dynd convention "out, in"
      for (intptr_t j = 1; j < nargs; ++j) {
        if (argtypes[j] != types[j - 1]) {
          matched = false;
          break;
        }
      }
      if (argtypes[0] != types[nargs - 1]) {
        matched = false;
      }
      // If we found a full match, return the kernel
      if (matched) {
        if (!uf->core_enabled) {
          ndt::type return_type = ndt_type_from_numpy_type_num(argtypes[0]);
          std::vector<ndt::type> param_types(nargs - 1);
          for (intptr_t j = 0; j < nargs - 1; ++j) {
            param_types[j] = ndt_type_from_numpy_type_num(argtypes[j + 1]);
          }
          nd::array af =
              nd::empty(ndt::make_funcproto(param_types, return_type));
          arrfunc_type_data *af_ptr = reinterpret_cast<arrfunc_type_data *>(
              af.get_readwrite_originptr());

          size_t out_af_size =
              sizeof(scalar_ufunc_data) + (nargs - 1) * sizeof(void *);
          void *data_raw = malloc(out_af_size);
          if (data_raw == NULL) {
            throw std::bad_alloc();
          }
          *af_ptr->get_data_as<void *>() = data_raw;
          memset(data_raw, 0, out_af_size);
          af_ptr->free_func = &delete_scalar_ufunc_data;
          af_ptr->instantiate = &instantiate_scalar_ufunc_ckernel;
          // Fill in the arrfunc instance data
          scalar_ufunc_data *data =
              reinterpret_cast<scalar_ufunc_data *>(data_raw);
          data->ufunc = uf;
          Py_INCREF(uf);
          data->param_count = nargs - 1;
          data->ckernel_acquires_gil = ckernel_acquires_gil;
          data->funcptr = uf->functions[i];
          data->ufunc_data = uf->data[i];
          af.flag_as_immutable();
          return wrap_array(af);
        }
        else {
          // TODO: support gufunc
          PyErr_SetString(PyExc_ValueError, "gufunc isn't implemented yet");
          return NULL;
        }
      }
    }

    PyErr_SetString(PyExc_ValueError,
                    "converting extended ufunc loops isn't implemented yet");
    return NULL;
  }
  catch (...) {
    translate_exception();
    return NULL;
  }
}

#endif // DYND_NUMPY_INTEROP
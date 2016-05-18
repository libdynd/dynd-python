//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>
#include <datetime.h>

#include <dynd/functional.hpp>
#include <dynd/kernels/tuple_assignment_kernels.hpp>

#include "assign.hpp"
#include "callables/assign_from_pyobject_callable.hpp"
#include "callables/assign_to_pyarrayobject_callable.hpp"
#include "callables/assign_to_pyobject_callable.hpp"

using namespace std;
using namespace dynd;

PYDYND_API void assign_init()
{
  typedef type_id_sequence<bool_id, int8_id, int16_id, int32_id, int64_id, int128_id, uint8_id, uint16_id, uint32_id,
                           uint64_id, uint128_id, float16_id, float32_id, float64_id, complex_float32_id,
                           complex_float64_id, bytes_id, fixed_bytes_id, string_id, fixed_string_id, option_id, type_id,
                           tuple_id, struct_id, fixed_dim_id, var_dim_id>
      type_ids;

  typedef type_sequence<bool, int8_t, int16_t, int32_t, int64_t, int128, uint8_t, uint16_t, uint32_t, uint64_t, uint128,
                        float16, float, double, dynd::complex<float>, dynd::complex<double>, bytes,
                        ndt::fixed_bytes_type, dynd::string, ndt::fixed_string_type, ndt::option_type, ndt::type,
                        ndt::tuple_type, ndt::struct_type, ndt::fixed_dim_type, ndt::var_dim_type>
      types;

  PyDateTime_IMPORT;

  for (const auto &pair : nd::callable::make_all<pydynd::nd::assign_from_pyobject_callable, types>()) {
    nd::assign.overload(pair.first[0], {ndt::make_type<pyobject_type>()}, pair.second);
  }
  for (const auto &pair : nd::callable::make_all<pydynd::nd::assign_to_pyobject_callable, type_ids>()) {
    nd::assign.overload(ndt::make_type<pyobject_type>(), {ndt::type(pair.first)}, pair.second);
  }
}

#if DYND_NUMPY_INTEROP

nd::callable assign_to_pyarrayobject = nd::functional::elwise(nd::make_callable<assign_to_pyarrayobject_callable>());

void array_copy_to_numpy(PyArrayObject *dst_arr, const dynd::ndt::type &src_tp, const char *src_arrmeta,
                         const char *src_data)
{
  intptr_t dst_ndim = PyArray_NDIM(dst_arr);
  intptr_t src_ndim = src_tp.get_ndim();
  uintptr_t dst_alignment = reinterpret_cast<uintptr_t>(PyArray_DATA(dst_arr));

  strided_of_numpy_arrmeta dst_am_holder;
  const char *dst_am = reinterpret_cast<const char *>(&dst_am_holder.sdt[NPY_MAXDIMS - dst_ndim]);
  // Fill in metadata for a multi-dim strided array, corresponding
  // to the numpy array, with a void type at the end for the numpy
  // specific data.
  for (intptr_t i = 0; i < dst_ndim; ++i) {
    dynd::fixed_dim_type_arrmeta &am = dst_am_holder.sdt[NPY_MAXDIMS - dst_ndim + i];
    am.stride = PyArray_STRIDE(dst_arr, (int)i);
    dst_alignment |= static_cast<uintptr_t>(am.stride);
    am.dim_size = PyArray_DIM(dst_arr, (int)i);
  }
  dynd::ndt::type dst_tp = dynd::ndt::make_type(dst_ndim, PyArray_SHAPE(dst_arr), dynd::ndt::make_type<void>());
  dst_am_holder.am.dst_dtype = PyArray_DTYPE(dst_arr);
  dst_am_holder.am.dst_alignment = dst_alignment;

  char *src_data_nonconst = const_cast<char *>(src_data);
  assign_to_pyarrayobject->call(dst_tp, dst_am, (char *)PyArray_DATA(dst_arr), 1, &src_tp, &src_arrmeta,
                                &src_data_nonconst, 1, NULL, std::map<std::string, dynd::ndt::type>());
}

#endif // DYND_NUMPY_INTEROP

//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>
#include <datetime.h>

#include <dynd/functional.hpp>
#include <dynd/kernels/tuple_assignment_kernels.hpp>

#include "assign.hpp"
#include "kernels/assign_from_pyobject_kernel.hpp"
#include "kernels/assign_to_pyarrayobject_kernel.hpp"
#include "kernels/assign_to_pyobject_kernel.hpp"

using namespace std;
using namespace dynd;

PYDYND_API void assign_init()
{
  typedef type_id_sequence<
      bool_type_id, int8_type_id, int16_type_id, int32_type_id, int64_type_id,
      int128_type_id, uint8_type_id, uint16_type_id, uint32_type_id,
      uint64_type_id, uint128_type_id, float16_type_id, float32_type_id,
      float64_type_id, complex_float32_type_id, complex_float64_type_id,
      bytes_type_id, fixed_bytes_type_id, string_type_id, fixed_string_type_id,
      date_type_id, time_type_id, datetime_type_id, option_type_id,
      type_type_id, tuple_type_id, struct_type_id, fixed_dim_type_id,
      var_dim_type_id> type_ids;

  PyDateTime_IMPORT;

  nd::callable &assign = nd::assign::get();
  for (const auto &pair :
       nd::callable::make_all<assign_from_pyobject_kernel, type_ids>()) {
    assign.set_overload(pair.first, {ndt::make_type<pyobject_type>()},
                        pair.second);
  }
  for (const auto &pair :
       nd::callable::make_all<assign_to_pyobject_kernel, type_ids>()) {
    assign.set_overload(ndt::make_type<pyobject_type>(),
                        {ndt::type(pair.first)}, pair.second);
  }
}

#if DYND_NUMPY_INTEROP

nd::callable assign_to_pyarrayobject::make()
{
  return nd::functional::elwise(
      nd::callable::make<assign_to_pyarrayobject_kernel>());
}

struct assign_to_pyarrayobject assign_to_pyarrayobject;

void array_copy_to_numpy(PyArrayObject *dst_arr, const dynd::ndt::type &src_tp,
                         const char *src_arrmeta, const char *src_data)
{
  intptr_t dst_ndim = PyArray_NDIM(dst_arr);
  intptr_t src_ndim = src_tp.get_ndim();
  uintptr_t dst_alignment = reinterpret_cast<uintptr_t>(PyArray_DATA(dst_arr));

  strided_of_numpy_arrmeta dst_am_holder;
  const char *dst_am = reinterpret_cast<const char *>(
      &dst_am_holder.sdt[NPY_MAXDIMS - dst_ndim]);
  // Fill in metadata for a multi-dim strided array, corresponding
  // to the numpy array, with a void type at the end for the numpy
  // specific data.
  for (intptr_t i = 0; i < dst_ndim; ++i) {
    dynd::fixed_dim_type_arrmeta &am =
        dst_am_holder.sdt[NPY_MAXDIMS - dst_ndim + i];
    am.stride = PyArray_STRIDE(dst_arr, (int)i);
    dst_alignment |= static_cast<uintptr_t>(am.stride);
    am.dim_size = PyArray_DIM(dst_arr, (int)i);
  }
  dynd::ndt::type dst_tp = dynd::ndt::make_type(
      dst_ndim, PyArray_SHAPE(dst_arr), dynd::ndt::make_type<void>());
  dst_am_holder.am.dst_dtype = PyArray_DTYPE(dst_arr);
  dst_am_holder.am.dst_alignment = dst_alignment;

  // TODO: This is a hack, need a proper way to pass this dst param
  intptr_t tmp_dst_arrmeta_size =
      dst_ndim * sizeof(dynd::fixed_dim_type_arrmeta) +
      sizeof(copy_to_numpy_arrmeta);
  dynd::nd::array tmp_dst(
      reinterpret_cast<dynd::array_preamble *>(
          dynd::make_array_memory_block(tmp_dst_arrmeta_size).get()),
      true);
  tmp_dst.get()->tp = dst_tp;
  tmp_dst.get()->flags =
      dynd::nd::read_access_flag | dynd::nd::write_access_flag;
  if (dst_tp.get_arrmeta_size() > 0) {
    memcpy(tmp_dst.get()->metadata(), dst_am, tmp_dst_arrmeta_size);
  }
  tmp_dst.get()->data = (char *)PyArray_DATA(dst_arr);
  char *src_data_nonconst = const_cast<char *>(src_data);
  assign_to_pyarrayobject::get()->call(
      tmp_dst.get_type(), tmp_dst.get()->metadata(), tmp_dst.data(), 1, &src_tp,
      &src_arrmeta, &src_data_nonconst, 1, NULL,
      std::map<std::string, dynd::ndt::type>());
}

#endif // DYND_NUMPY_INTEROP

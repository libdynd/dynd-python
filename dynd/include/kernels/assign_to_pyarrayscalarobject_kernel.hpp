//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <dynd/kernels/base_kernel.hpp>

#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>

#include "types/pyobject_type.hpp"

using namespace std;
using namespace dynd;

template <type_id_t Arg0ID>
struct assign_to_pyarrayscalarobject_kernel;

template <>
struct assign_to_pyarrayscalarobject_kernel<bool_type_id>
    : nd::base_kernel<assign_to_pyarrayscalarobject_kernel<bool_type_id>, 1> {
  void single(char *res, char *const *args)
  {
    if (*reinterpret_cast<bool *>(args[0])) {
      Py_INCREF(PyArrayScalar_True);
      *reinterpret_cast<PyObject **>(res) = PyArrayScalar_True;
    }
    else {
      Py_INCREF(PyArrayScalar_False);
      *reinterpret_cast<PyObject **>(res) = PyArrayScalar_False;
    }
  }
};

#define assign_to_pyarrayscalarobject_kernel(ARG0_ID, CLS)                     \
  template <>                                                                  \
  struct assign_to_pyarrayscalarobject_kernel<ARG0_ID>                         \
      : nd::base_kernel<assign_to_pyarrayscalarobject_kernel<ARG0_ID>, 1> {    \
    typedef typename type_of<ARG0_ID>::type arg0_type;                         \
                                                                               \
    void single(char *res, char *const *args)                                  \
    {                                                                          \
      *reinterpret_cast<PyObject **>(res) = PyArrayScalar_New(CLS);            \
      PyArrayScalar_ASSIGN(*reinterpret_cast<PyObject **>(res), CLS,           \
                           *reinterpret_cast<arg0_type *>(args[0]));           \
    }                                                                          \
  }

assign_to_pyarrayscalarobject_kernel(int8_type_id, Int8);
assign_to_pyarrayscalarobject_kernel(int16_type_id, Int16);
assign_to_pyarrayscalarobject_kernel(int32_type_id, Int32);
assign_to_pyarrayscalarobject_kernel(int64_type_id, Int64);
assign_to_pyarrayscalarobject_kernel(uint8_type_id, UInt8);
assign_to_pyarrayscalarobject_kernel(uint16_type_id, UInt16);
assign_to_pyarrayscalarobject_kernel(uint32_type_id, UInt32);
assign_to_pyarrayscalarobject_kernel(uint64_type_id, UInt64);
assign_to_pyarrayscalarobject_kernel(float32_type_id, Float32);
assign_to_pyarrayscalarobject_kernel(float64_type_id, Float64);

#undef assign_to_pyarrayscalarobject_kernel

template <>
struct assign_to_pyarrayscalarobject_kernel<complex_float32_type_id>
    : nd::base_kernel<
          assign_to_pyarrayscalarobject_kernel<complex_float32_type_id>, 1> {
  void single(char *res, char *const *args)
  {
    *reinterpret_cast<PyObject **>(res) = PyArrayScalar_New(Complex64);
    PyArrayScalar_VAL(*reinterpret_cast<PyObject **>(res), Complex64).real =
        reinterpret_cast<dynd::complex<float> *>(args[0])->real();
    PyArrayScalar_VAL(*reinterpret_cast<PyObject **>(res), Complex64).imag =
        reinterpret_cast<dynd::complex<float> *>(args[0])->imag();
  }
};

template <>
struct assign_to_pyarrayscalarobject_kernel<complex_float64_type_id>
    : nd::base_kernel<
          assign_to_pyarrayscalarobject_kernel<complex_float64_type_id>, 1> {
  void single(char *res, char *const *args)
  {
    *reinterpret_cast<PyObject **>(res) = PyArrayScalar_New(Complex128);
    PyArrayScalar_VAL(*reinterpret_cast<PyObject **>(res), Complex128).real =
        reinterpret_cast<dynd::complex<double> *>(args[0])->real();
    PyArrayScalar_VAL(*reinterpret_cast<PyObject **>(res), Complex128).imag =
        reinterpret_cast<dynd::complex<double> *>(args[0])->imag();
  }
};

namespace dynd {
namespace ndt {

  template <type_id_t Arg0ID>
  struct traits<assign_to_pyarrayscalarobject_kernel<Arg0ID>> {
    static type equivalent()
    {
      return callable_type::make(make_type<pyobject_type>(), {type(Arg0ID)});
    }
  };

} // namespace dynd::ndt
} // namespace dynd

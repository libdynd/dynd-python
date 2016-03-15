//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include <Python.h>

#include <type_traits>

#include <dynd/type_registry.hpp>

#include "types/pyobject_type.hpp"

using namespace dynd;

// Unpack and convert a single element to a PyObject.
template <typename T,
          typename std::enable_if<
              std::is_integral<T>::value && std::is_signed<T>::value && sizeof(T) <= sizeof(long long), int>::type = 0>
inline PyObject *unpack_single(const char *data)
{
  return PyLong_FromLongLong(*reinterpret_cast<const T *>(data));
}

template <typename T, typename std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value &&
                                                    sizeof(T) <= sizeof(unsigned long long),
                                                int> = 0>
inline PyObject *unpack_single(const char *data)
{
  return PyLong_FromUnsignedLongLong(*reinterpret_cast<const T *>(data));
}

template <typename T, typename std::enable_if_t<std::is_same<T, bool1>::value, int> = 0>
inline PyObject *unpack_single(const char *data)
{
  return PyBool_FromLong(*reinterpret_cast<const T *>(data));
}

template <typename T, typename std::enable_if_t<std::is_same<T, float64>::value, int> = 0>
inline PyObject *unpack_single(const char *data)
{
  return PyFloat_FromDouble(*reinterpret_cast<const T *>(data));
}

template <typename T, typename std::enable_if_t<std::is_same<T, complex128>::value, int> = 0>
inline PyObject *unpack_single(const char *data)
{
  complex128 c = *reinterpret_cast<const T *>(data);
  return PyComplex_FromDoubles(c.real(), c.imag());
}

template <typename T, typename std::enable_if_t<std::is_same<T, std::string>::value, int> = 0>
inline PyObject *unpack_single(const char *data)
{
  const std::string &s = *reinterpret_cast<const T *>(data);
  return PyUnicode_FromString(s.data());
}

template <typename T, typename std::enable_if_t<std::is_same<T, ndt::type>::value, int> = 0>
inline PyObject *unpack_single(const char *data)
{
  return pydynd::type_from_cpp(*reinterpret_cast<const T *>(data));
}

// Convert a single element to a PyObject.
template <typename T,
          typename std::enable_if_t<
              std::is_integral<T>::value && std::is_signed<T>::value && sizeof(T) <= sizeof(long long), int> = 0>
inline PyObject *convert_single(T value)
{
  return PyLong_FromLongLong(value);
}

template <typename T, typename std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value &&
                                                    sizeof(T) <= sizeof(unsigned long long),
                                                int> = 0>
inline PyObject *convert_single(T value)
{
  return PyLong_FromUnsignedLongLong(value);
}

template <typename T, typename std::enable_if_t<std::is_same<T, bool1>::value, int> = 0>
inline PyObject *convert_single(T value)
{
  return PyBool_FromLong(value);
}

template <typename T, typename std::enable_if_t<std::is_same<T, float64>::value, int> = 0>
inline PyObject *convert_single(T value)
{
  return PyFloat_FromDouble(value);
}

template <typename T, typename std::enable_if_t<std::is_same<T, complex128>::value, int> = 0>
inline PyObject *convert_single(T value)
{
  return PyComplex_FromDoubles(value.real(), value.imag());
}

template <typename T, typename std::enable_if_t<std::is_same<T, std::string>::value, int> = 0>
inline PyObject *convert_single(const T &value)
{
  return PyUnicode_FromString(value.data());
}

template <typename T, typename std::enable_if_t<std::is_same<T, ndt::type>::value, int> = 0>
inline PyObject *convert_single(const T &value)
{
  return pydynd::type_from_cpp(value);
}

template <typename T>
PyObject *unpack_vector(const char *data)
{
  auto &vec = *reinterpret_cast<const std::vector<T> *>(data);
  PyObject *lst, *item;

  lst = PyList_New(vec.size());
  if (lst == NULL) {
    return NULL;
  }

  for (size_t i = 0; i < vec.size(); ++i) {
    item = convert_single<T>(vec[i]);
    if (item == NULL) {
      Py_DECREF(lst);
      return NULL;
    }
    PyList_SET_ITEM(lst, i, item);
  }

  return lst;
}

template <typename T>
inline PyObject *unpack(bool is_vector, const char *data)
{
  if (is_vector) {
    return unpack_vector<T>(data);
  }
  else {
    return unpack_single<T>(data);
  }
}

PyObject *from_type_property(const std::pair<ndt::type, const char *> &pair)
{
  type_id_t id = pair.first.get_id();
  bool is_vector = false;

  if (id == fixed_dim_id) {
    id = pair.first.get_dtype().get_id();
    is_vector = true;
  }

  switch (id) {
  case bool_id:
    return unpack<bool1>(is_vector, pair.second);
  case int8_id:
    return unpack<int8>(is_vector, pair.second);
  case int16_id:
    return unpack<int16>(is_vector, pair.second);
  case int32_id:
    return unpack<int32>(is_vector, pair.second);
  case int64_id:
    return unpack<int64>(is_vector, pair.second);
  case uint8_id:
    return unpack<uint8>(is_vector, pair.second);
  case uint16_id:
    return unpack<uint16>(is_vector, pair.second);
  case uint32_id:
    return unpack<uint32>(is_vector, pair.second);
  case uint64_id:
    return unpack<uint64>(is_vector, pair.second);
  case float64_id:
    return unpack<float64>(is_vector, pair.second);
  case complex_float64_id:
    return unpack<complex128>(is_vector, pair.second);
  case type_id:
    return unpack<ndt::type>(is_vector, pair.second);
  case string_id:
    return unpack<std::string>(is_vector, pair.second);
  default:
    throw std::runtime_error("invalid type property");
  }
}


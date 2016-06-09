//
// Copyright (C) 2011-16 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// This header defines some functions to
// interoperate with numpy
//

#include "numpy_type_interop.hpp"

#if DYND_NUMPY_INTEROP

#include <dynd/types/struct_type.hpp>

#include <numpy/arrayscalars.h>

using namespace std;

PyArray_Descr *pydynd::numpy_dtype_from__type(const dynd::ndt::type &tp)
{
  switch (tp.get_id()) {
  case dynd::bool_id:
    return PyArray_DescrFromType(NPY_BOOL);
  case dynd::int8_id:
    return PyArray_DescrFromType(NPY_INT8);
  case dynd::int16_id:
    return PyArray_DescrFromType(NPY_INT16);
  case dynd::int32_id:
    return PyArray_DescrFromType(NPY_INT32);
  case dynd::int64_id:
    return PyArray_DescrFromType(NPY_INT64);
  case dynd::uint8_id:
    return PyArray_DescrFromType(NPY_UINT8);
  case dynd::uint16_id:
    return PyArray_DescrFromType(NPY_UINT16);
  case dynd::uint32_id:
    return PyArray_DescrFromType(NPY_UINT32);
  case dynd::uint64_id:
    return PyArray_DescrFromType(NPY_UINT64);
  case dynd::float32_id:
    return PyArray_DescrFromType(NPY_FLOAT32);
  case dynd::float64_id:
    return PyArray_DescrFromType(NPY_FLOAT64);
  case dynd::complex_float32_id:
    return PyArray_DescrFromType(NPY_CFLOAT);
  case dynd::complex_float64_id:
    return PyArray_DescrFromType(NPY_CDOUBLE);
  case dynd::fixed_string_id: {
    const dynd::ndt::fixed_string_type *ftp = tp.extended<dynd::ndt::fixed_string_type>();
    PyArray_Descr *result;
    switch (ftp->get_encoding()) {
    case dynd::string_encoding_ascii:
      result = PyArray_DescrNewFromType(NPY_STRING);
      result->elsize = (int)ftp->get_data_size();
      return result;
    case dynd::string_encoding_utf_32:
      result = PyArray_DescrNewFromType(NPY_UNICODE);
      result->elsize = (int)ftp->get_data_size();
      return result;
    default:
      break;
    }
    break;
  }
  /*
        case tuple_id: {
            const tuple_type *ttp = tp.extended<tuple_type>();
            const vector<ndt::type>& fields = ttp->get_fields();
            size_t num_fields = fields.size();
            const vector<size_t>& offsets = ttp->get_offsets();
            // TODO: Deal with the names better
            py_ref names_obj = capture_if_not_null(PyList_New(num_fields));
            for (size_t i = 0; i < num_fields; ++i) {
                stringstream ss;
                ss << "f" << i;
                PyList_SET_ITEM(names_obj.get(), i,
PyString_FromString(ss.str().c_str()));
            }
            py_ref formats_obj = capture_if_not_null(PyList_New(num_fields));
            for (size_t i = 0; i < num_fields; ++i) {
                PyList_SET_ITEM(formats_obj.get(), i, (PyObject
*)numpy_dtype_from__type(fields[i]));
            }
            py_ref offsets_obj = capture_if_not_null(PyList_New(num_fields));
            for (size_t i = 0; i < num_fields; ++i) {
                PyList_SET_ITEM(offsets_obj.get(), i,
PyLong_FromSize_t(offsets[i]));
            }
            py_ref itemsize_obj = capture_if_not_null(PyLong_FromSize_t(tp.get_data_size()));
            py_ref dict_obj = capture_if_not_null(PyDict_New());
            PyDict_SetItemString(dict_obj.get(), "names", names_obj.get());
            PyDict_SetItemString(dict_obj.get(), "formats", formats_obj.get());
            PyDict_SetItemString(dict_obj.get(), "offsets", offsets_obj.get());
            PyDict_SetItemString(dict_obj.get(), "itemsize", itemsize_obj.get());
            PyArray_Descr *result = NULL;
            if (PyArray_DescrConverter(dict_obj, &result) != NPY_SUCCEED) {
                throw dynd::type_error("failed to convert tuple dtype into numpy
dtype via dict");
            }
            return result;
        }
        case struct_id: {
            const struct_type *ttp = tp.extended<struct_type>();
            size_t field_count = ttp->get_field_count();
            size_t max_numpy_alignment = 1;
            std::vector<uintptr_t> offsets(field_count);
            struct_type::fill_default_data_offsets(field_count,
ttp->get_field_types_raw(), offsets.data());
            py_ref names_obj = capture_if_not_null(PyList_New(field_count));
            for (size_t i = 0; i < field_count; ++i) {
                const string_type_data& fname = ttp->get_field_name(i);
#if PY_VERSION_HEX >= 0x03000000
                py_ref name_str = capture_if_not_null(PyUnicode_FromStringAndSize(
                    fname.begin, fname.end - fname.begin));
#else
                py_ref name_str = capture_if_not_null(PyString_FromStringAndSize(
                    fname.begin, fname.end - fname.begin));
#endif
                PyList_SET_ITEM(names_obj.get(), i, release(std::move(name_str)));
            }
            py_ref formats_obj = capture_if_not_null(PyList_New(field_count));
            for (size_t i = 0; i < field_count; ++i) {
                PyArray_Descr *npdt =
numpy_dtype_from__type(ttp->get_field_type(i));
                max_numpy_alignment = max(max_numpy_alignment,
(size_t)npdt->alignment);
                PyList_SET_ITEM((PyObject *)formats_obj, i, (PyObject *)npdt);
            }
            py_ref offsets_obj = capture_if_not_null(PyList_New(field_count));
            for (size_t i = 0; i < field_count; ++i) {
                PyList_SET_ITEM((PyObject *)offsets_obj, i,
PyLong_FromSize_t(offsets[i]));
            }
            py_ref itemsize_obj =
                capture_if_not_null(PyLong_FromSize_t(tp.get_default_data_size()));
            py_ref dict_obj = capture_if_not_null(PyDict_New());
            PyDict_SetItemString(dict_obj.get(), "names", names_obj.get());
            PyDict_SetItemString(dict_obj.get(), "formats", formats_obj.get());
            PyDict_SetItemString(dict_obj.get(), "offsets", offsets_obj.get());
            PyDict_SetItemString(dict_obj.get(), "itemsize", itemsize_obj.get());
            // This isn't quite right, but the rules between numpy and dynd
            // differ enough to make this tricky.
            if (max_numpy_alignment > 1 &&
                            max_numpy_alignment == tp.get_data_alignment()) {
                Py_INCREF(Py_True);
                PyDict_SetItemString(dict_obj.get(), "aligned", Py_True);
            }
            PyArray_Descr *result = NULL;
            if (PyArray_DescrConverter(dict_obj.get(), &result) != NPY_SUCCEED) {
                stringstream ss;
                ss << "failed to convert dtype " << tp << " into numpy dtype via
dict";
                throw dynd::type_error(ss.str());
            }
            return result;
        }
        case fixed_dim_id: {
            ndt::type child_tp = tp;
            vector<intptr_t> shape;
            do {
                const cfixed_dim_type *ttp =
child_tp.extended<cfixed_dim_type>();
                shape.push_back(ttp->get_fixed_dim_size());
                if (child_tp.get_data_size() !=
ttp->get_element_type().get_data_size() * shape.back()) {
                    stringstream ss;
                    ss << "Cannot convert dynd type " << tp << " into a numpy
dtype because it is not C-order";
                    throw dynd::type_error(ss.str());
                }
                child_tp = ttp->get_element_type();
            } while (child_tp.get_id() == cfixed_dim_id);
            py_ref dtype_obj = capture_if_not_null((PyObject
*)numpy_dtype_from__type(child_tp));
            py_ref shape_obj = capture_if_not_null(intptr_array_as_tuple((int)shape.size(),
&shape[0]));
            py_ref tuple_obj = capture_if_not_null(PyTuple_New(2));
            PyTuple_SET_ITEM(tuple_obj.get(), 0, release(std::move(dtype_obj)));
            PyTuple_SET_ITEM(tuple_obj.get(), 1, release(std::move(shape_obj)));
            PyArray_Descr *result = NULL;
            if (PyArray_DescrConverter(tuple_obj, &result) != NPY_SUCCEED) {
                throw dynd::type_error("failed to convert dynd type into numpy
subarray dtype");
            }
            return result;
        }
*/
  default:
    break;
  }

  stringstream ss;
  ss << "cannot convert dynd type " << tp << " into a Numpy dtype";
  throw dynd::type_error(ss.str());
}

PyArray_Descr *pydynd::numpy_dtype_from__type(const dynd::ndt::type &tp, const char *arrmeta)
{
  switch (tp.get_id()) {
  case dynd::struct_id: {
    throw std::runtime_error("converting");
    if (arrmeta == NULL) {
      stringstream ss;
      ss << "Can only convert dynd type " << tp << " into a numpy dtype with array arrmeta";
      throw dynd::type_error(ss.str());
    }
    const dynd::ndt::struct_type *stp = tp.extended<dynd::ndt::struct_type>();
    const uintptr_t *arrmeta_offsets = stp->get_arrmeta_offsets_raw();
    const uintptr_t *offsets = reinterpret_cast<const uintptr_t *>(arrmeta);
    size_t field_count = stp->get_field_count();
    size_t max_numpy_alignment = 1;

    py_ref names_obj = capture_if_not_null(PyList_New(field_count));
    for (size_t i = 0; i < field_count; ++i) {
      const dynd::string &fname = stp->get_field_name(i);
#if PY_VERSION_HEX >= 0x03000000
      py_ref name_str = capture_if_not_null(PyUnicode_FromStringAndSize(fname.begin(), fname.end() - fname.begin()));
#else
      py_ref name_str = capture_if_not_null(PyString_FromStringAndSize(fname.begin(), fname.end() - fname.begin()));
#endif
      PyList_SET_ITEM(names_obj.get(), i, release(std::move(name_str)));
    }

    py_ref formats_obj = capture_if_not_null(PyList_New(field_count));
    for (size_t i = 0; i < field_count; ++i) {
      PyArray_Descr *npdt = numpy_dtype_from__type(stp->get_field_type(i), arrmeta + arrmeta_offsets[i]);
      max_numpy_alignment = max(max_numpy_alignment, (size_t)npdt->alignment);
      PyList_SET_ITEM(formats_obj.get(), i, (PyObject *)npdt);
    }

    py_ref offsets_obj = capture_if_not_null(PyList_New(field_count));
    for (size_t i = 0; i < field_count; ++i) {
      PyList_SET_ITEM(offsets_obj.get(), i, PyLong_FromSize_t(offsets[i]));
    }

    py_ref itemsize_obj = capture_if_not_null(PyLong_FromSize_t(tp.get_data_size()));

    py_ref dict_obj = capture_if_not_null(PyDict_New());
    PyDict_SetItemString(dict_obj.get(), "names", names_obj.get());
    PyDict_SetItemString(dict_obj.get(), "formats", formats_obj.get());
    PyDict_SetItemString(dict_obj.get(), "offsets", offsets_obj.get());
    PyDict_SetItemString(dict_obj.get(), "itemsize", itemsize_obj.get());
    // This isn't quite right, but the rules between numpy and dynd
    // differ enough to make this tricky.
    if (max_numpy_alignment > 1 && max_numpy_alignment == tp.get_data_alignment()) {
      Py_INCREF(Py_True);
      PyDict_SetItemString(dict_obj.get(), "aligned", Py_True);
    }

    PyArray_Descr *result = NULL;
    if (PyArray_DescrConverter(dict_obj.get(), &result) != NPY_SUCCEED) {
      throw dynd::type_error("failed to convert dtype into numpy struct dtype via dict");
    }
    return result;
  }
  default:
    return numpy_dtype_from__type(tp);
  }
}

int pydynd::_type_from_numpy_scalar_typeobject(PyTypeObject *obj, dynd::ndt::type &out_d)
{
  if (obj == &PyBoolArrType_Type) {
    out_d = dynd::ndt::make_type<dynd::bool1>();
  }
  else if (obj == &PyByteArrType_Type) {
    out_d = dynd::ndt::make_type<npy_byte>();
  }
  else if (obj == &PyUByteArrType_Type) {
    out_d = dynd::ndt::make_type<npy_ubyte>();
  }
  else if (obj == &PyShortArrType_Type) {
    out_d = dynd::ndt::make_type<npy_short>();
  }
  else if (obj == &PyUShortArrType_Type) {
    out_d = dynd::ndt::make_type<npy_ushort>();
  }
  else if (obj == &PyIntArrType_Type) {
    out_d = dynd::ndt::make_type<npy_int>();
  }
  else if (obj == &PyUIntArrType_Type) {
    out_d = dynd::ndt::make_type<npy_uint>();
  }
  else if (obj == &PyLongArrType_Type) {
    out_d = dynd::ndt::make_type<npy_long>();
  }
  else if (obj == &PyULongArrType_Type) {
    out_d = dynd::ndt::make_type<npy_ulong>();
  }
  else if (obj == &PyLongLongArrType_Type) {
    out_d = dynd::ndt::make_type<npy_longlong>();
  }
  else if (obj == &PyULongLongArrType_Type) {
    out_d = dynd::ndt::make_type<npy_ulonglong>();
  }
  else if (obj == &PyFloatArrType_Type) {
    out_d = dynd::ndt::make_type<npy_float>();
  }
  else if (obj == &PyDoubleArrType_Type) {
    out_d = dynd::ndt::make_type<npy_double>();
  }
  else if (obj == &PyCFloatArrType_Type) {
    out_d = dynd::ndt::make_type<dynd::complex<float>>();
  }
  else if (obj == &PyCDoubleArrType_Type) {
    out_d = dynd::ndt::make_type<dynd::complex<double>>();
  }
  else {
    return -1;
  }

  return 0;
}

dynd::ndt::type pydynd::_type_of_numpy_scalar(PyObject *obj)
{
  if (PyArray_IsScalar(obj, Bool)) {
    return dynd::ndt::make_type<dynd::bool1>();
  }
  else if (PyArray_IsScalar(obj, Byte)) {
    return dynd::ndt::make_type<npy_byte>();
  }
  else if (PyArray_IsScalar(obj, UByte)) {
    return dynd::ndt::make_type<npy_ubyte>();
  }
  else if (PyArray_IsScalar(obj, Short)) {
    return dynd::ndt::make_type<npy_short>();
  }
  else if (PyArray_IsScalar(obj, UShort)) {
    return dynd::ndt::make_type<npy_ushort>();
  }
  else if (PyArray_IsScalar(obj, Int)) {
    return dynd::ndt::make_type<npy_int>();
  }
  else if (PyArray_IsScalar(obj, UInt)) {
    return dynd::ndt::make_type<npy_uint>();
  }
  else if (PyArray_IsScalar(obj, Long)) {
    return dynd::ndt::make_type<npy_long>();
  }
  else if (PyArray_IsScalar(obj, ULong)) {
    return dynd::ndt::make_type<npy_ulong>();
  }
  else if (PyArray_IsScalar(obj, LongLong)) {
    return dynd::ndt::make_type<npy_longlong>();
  }
  else if (PyArray_IsScalar(obj, ULongLong)) {
    return dynd::ndt::make_type<npy_ulonglong>();
  }
  else if (PyArray_IsScalar(obj, Float)) {
    return dynd::ndt::make_type<float>();
  }
  else if (PyArray_IsScalar(obj, Double)) {
    return dynd::ndt::make_type<double>();
  }
  else if (PyArray_IsScalar(obj, CFloat)) {
    return dynd::ndt::make_type<dynd::complex<float>>();
  }
  else if (PyArray_IsScalar(obj, CDouble)) {
    return dynd::ndt::make_type<dynd::complex<double>>();
  }

  throw dynd::type_error("could not deduce a pydynd type from the numpy scalar object");
}

dynd::ndt::type pydynd::array_from_numpy_array2(PyArrayObject *obj)
{
  PyArray_Descr *dtype = PyArray_DESCR(obj);

  if (PyDataType_FLAGCHK(dtype, NPY_ITEM_HASOBJECT)) {
    return dynd::ndt::make_type(PyArray_NDIM(obj), PyArray_SHAPE(obj),
                                pydynd::_type_from_numpy_dtype(dtype).get_canonical_type());
  }
  else {
    // Get the dtype of the array
    dynd::ndt::type d = pydynd::_type_from_numpy_dtype(PyArray_DESCR(obj), get_alignment_of(obj));
    return dynd::ndt::make_type(PyArray_NDIM(obj), PyArray_DIMS(obj), d);
  }
}

dynd::ndt::type pydynd::array_from_numpy_scalar2(PyObject *obj)
{
  if (PyArray_IsScalar(obj, Bool)) {
    return dynd::ndt::make_type<bool>();
  }

  if (PyArray_IsScalar(obj, Byte)) {
    return dynd::ndt::make_type<signed char>();
  }

  if (PyArray_IsScalar(obj, UByte)) {
    return dynd::ndt::make_type<unsigned char>();
  }

  if (PyArray_IsScalar(obj, Short)) {
    return dynd::ndt::make_type<short>();
  }

  if (PyArray_IsScalar(obj, UShort)) {
    return dynd::ndt::make_type<unsigned short>();
  }

  if (PyArray_IsScalar(obj, Int)) {
    return dynd::ndt::make_type<int>();
  }

  if (PyArray_IsScalar(obj, UInt)) {
    return dynd::ndt::make_type<unsigned int>();
  }

  if (PyArray_IsScalar(obj, Long)) {
    return dynd::ndt::make_type<long>();
  }

  if (PyArray_IsScalar(obj, ULong)) {
    return dynd::ndt::make_type<unsigned long>();
  }

  if (PyArray_IsScalar(obj, LongLong)) {
    return dynd::ndt::make_type<long long>();
  }

  if (PyArray_IsScalar(obj, ULongLong)) {
    return dynd::ndt::make_type<unsigned long long>();
  }

  if (PyArray_IsScalar(obj, Float)) {
    return dynd::ndt::make_type<float>();
  }

  if (PyArray_IsScalar(obj, Double)) {
    return dynd::ndt::make_type<double>();
  }

  if (PyArray_IsScalar(obj, CFloat)) {
    return dynd::ndt::make_type<dynd::complex<float>>();
  }

  if (PyArray_IsScalar(obj, CDouble)) {
    return dynd::ndt::make_type<dynd::complex<double>>();
  }

  if (PyArray_IsScalar(obj, Void)) {
    return dynd::ndt::make_type<void>();
  }

  stringstream ss;
  py_ref obj_tp = capture_if_not_null(PyObject_Repr((PyObject *)Py_TYPE(obj)));
  ss << "could not create a dynd array from the numpy scalar object";
  ss << " of type " << pydynd::pystring_as_string(obj_tp.get());
  throw dynd::type_error(ss.str());
}

bool pydynd::is_numpy_dtype(PyObject *o) { return PyArray_DescrCheck(o); }

#endif

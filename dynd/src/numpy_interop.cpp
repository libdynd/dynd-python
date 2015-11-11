//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "numpy_interop.hpp"

#if DYND_NUMPY_INTEROP

#include <dynd/types/byteswap_type.hpp>
#include <dynd/types/view_type.hpp>
#include <dynd/types/type_alignment.hpp>
#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/memblock/external_memory_block.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/datetime_type.hpp>
#include <dynd/types/property_type.hpp>
#include <dynd/types/adapt_type.hpp>

#include "type_functions.hpp"
#include "array_functions.hpp"
#include "utility_functions.hpp"
#include "copy_from_numpy_arrfunc.hpp"

#include <numpy/arrayscalars.h>

using namespace std;

void pydynd::extract_fields_from_numpy_struct(
    PyArray_Descr *d, vector<PyArray_Descr *> &out_field_dtypes,
    vector<string> &out_field_names, vector<size_t> &out_field_offsets)
{
  if (!PyDataType_HASFIELDS(d)) {
    throw dynd::type_error(
        "Tried to treat a non-structured NumPy dtype as a structure");
  }

  PyObject *names = d->names;
  Py_ssize_t names_size = PyTuple_GET_SIZE(names);

  for (Py_ssize_t i = 0; i < names_size; ++i) {
    PyObject *key = PyTuple_GET_ITEM(names, i);
    PyObject *tup = PyDict_GetItem(d->fields, key);
    PyArray_Descr *fld_dtype;
    PyObject *title;
    int offset = 0;
    if (!PyArg_ParseTuple(tup, "Oi|O", &fld_dtype, &offset, &title)) {
      throw dynd::type_error("Numpy struct dtype has corrupt data");
    }
    out_field_dtypes.push_back(fld_dtype);
    out_field_names.push_back(pystring_as_string(key));
    out_field_offsets.push_back(offset);
  }
}

dynd::ndt::type make_struct_type_from_numpy_struct(PyArray_Descr *d,
                                                   size_t data_alignment)
{
  vector<PyArray_Descr *> field_dtypes;
  vector<string> field_names;
  vector<size_t> field_offsets;

  pydynd::extract_fields_from_numpy_struct(d, field_dtypes, field_names,
                                           field_offsets);

  vector<dynd::ndt::type> field_types;

  if (data_alignment == 0) {
    data_alignment = (size_t)d->alignment;
  }

  // The alignment must divide into the total element size,
  // shrink it until it does.
  while (!dynd::offset_is_aligned((size_t)d->elsize, data_alignment)) {
    data_alignment >>= 1;
  }

  for (size_t i = 0; i < field_dtypes.size(); ++i) {
    PyArray_Descr *fld_dtype = field_dtypes[i];
    size_t offset = field_offsets[i];
    field_types.push_back(
        pydynd::_type_from_numpy_dtype(fld_dtype, data_alignment));
    // If the field isn't aligned enough, turn it into an unaligned type
    if (!dynd::offset_is_aligned(offset | data_alignment,
                                 field_types.back().get_data_alignment())) {
      field_types.back() = make_unaligned(field_types.back());
    }
  }

  // Make a cstruct if possible, struct otherwise
  return dynd::ndt::struct_type::make(field_names, field_types);
}

dynd::ndt::type pydynd::_type_from_numpy_dtype(PyArray_Descr *d,
                                               size_t data_alignment)
{
  dynd::ndt::type dt;

  if (d->subarray) {
    dt = _type_from_numpy_dtype(d->subarray->base, data_alignment);
    return dynd_make_fixed_dim_type(d->subarray->shape, dt);
  }

  switch (d->type_num) {
  case NPY_BOOL:
    dt = dynd::ndt::type::make<dynd::bool1>();
    break;
  case NPY_BYTE:
    dt = dynd::ndt::type::make<npy_byte>();
    break;
  case NPY_UBYTE:
    dt = dynd::ndt::type::make<npy_ubyte>();
    break;
  case NPY_SHORT:
    dt = dynd::ndt::type::make<npy_short>();
    break;
  case NPY_USHORT:
    dt = dynd::ndt::type::make<npy_ushort>();
    break;
  case NPY_INT:
    dt = dynd::ndt::type::make<npy_int>();
    break;
  case NPY_UINT:
    dt = dynd::ndt::type::make<npy_uint>();
    break;
  case NPY_LONG:
    dt = dynd::ndt::type::make<npy_long>();
    break;
  case NPY_ULONG:
    dt = dynd::ndt::type::make<npy_ulong>();
    break;
  case NPY_LONGLONG:
    dt = dynd::ndt::type::make<npy_longlong>();
    break;
  case NPY_ULONGLONG:
    dt = dynd::ndt::type::make<npy_ulonglong>();
    break;
  case NPY_FLOAT:
    dt = dynd::ndt::type::make<float>();
    break;
  case NPY_DOUBLE:
    dt = dynd::ndt::type::make<double>();
    break;
  case NPY_CFLOAT:
    dt = dynd::ndt::type::make<dynd::complex<float>>();
    break;
  case NPY_CDOUBLE:
    dt = dynd::ndt::type::make<dynd::complex<double>>();
    break;
  case NPY_STRING:
    dt = dynd::ndt::fixed_string_type::make(d->elsize,
                                            dynd::string_encoding_ascii);
    break;
  case NPY_UNICODE:
    dt = dynd::ndt::fixed_string_type::make(d->elsize / 4,
                                            dynd::string_encoding_utf_32);
    break;
  case NPY_VOID:
    dt = make_struct_type_from_numpy_struct(d, data_alignment);
    break;
  case NPY_OBJECT: {
    if (d->fields != NULL && d->fields != Py_None) {
      // Check for an h5py vlen string type (h5py 2.2 style)
      PyObject *vlen_tup = PyMapping_GetItemString(d->fields, "vlen");
      if (vlen_tup != NULL) {
        pyobject_ownref vlen_tup_owner(vlen_tup);
        if (PyTuple_Check(vlen_tup) && PyTuple_GET_SIZE(vlen_tup) == 3) {
          PyObject *type_dict = PyTuple_GET_ITEM(vlen_tup, 2);
          if (PyDict_Check(type_dict)) {
            PyObject *type = PyDict_GetItemString(type_dict, "type");
#if PY_VERSION_HEX < 0x03000000
            if (type == (PyObject *)&PyString_Type ||
                type == (PyObject *)&PyUnicode_Type) {
#else
            if (type == (PyObject *)&PyUnicode_Type) {
#endif
              dt = dynd::ndt::string_type::make();
            }
          }
        }
      } else {
        PyErr_Clear();
      }
    }
    // Check for an h5py vlen string type (h5py 2.3 style)
    if (d->metadata != NULL && PyDict_Check(d->metadata)) {
      PyObject *type = PyDict_GetItemString(d->metadata, "vlen");
#if PY_VERSION_HEX < 0x03000000
      if (type == (PyObject *)&PyString_Type ||
          type == (PyObject *)&PyUnicode_Type) {
#else
      if (type == (PyObject *)&PyUnicode_Type) {
#endif
        dt = dynd::ndt::string_type::make();
      }
    }
    break;
  }
#if NPY_API_VERSION >= 6 // At least NumPy 1.6
  case NPY_DATETIME: {
    // Get the dtype info through the CPython API, slower
    // but lets NumPy's datetime API change without issue.
    pyobject_ownref mod(PyImport_ImportModule("numpy"));
    pyobject_ownref dd(PyObject_CallMethod(mod.get(),
                                           const_cast<char *>("datetime_data"),
                                           const_cast<char *>("O"), d));
    PyObject *unit = PyTuple_GetItem(dd.get(), 0);
    if (unit == NULL) {
      throw runtime_error("");
    }
    std::string s = pydynd::pystring_as_string(unit);
    if (s == "D") {
      // If it's 'datetime64[D]', then use an adapter type with appropriate
      // metadata
      dt = dynd::ndt::adapt_type::make(dynd::ndt::type::make<int64_t>(),
                                       dynd::ndt::date_type::make(),
                                       "days since 1970");
    } else if (s == "h") {
      dt = dynd::ndt::adapt_type::make(
          dynd::ndt::type::make<int64_t>(),
          dynd::ndt::datetime_type::make(dynd::tz_utc), "hours since 1970");
    } else if (s == "m") {
      dt = dynd::ndt::adapt_type::make(
          dynd::ndt::type::make<int64_t>(),
          dynd::ndt::datetime_type::make(dynd::tz_utc), "minutes since 1970");
    } else if (s == "s") {
      dt = dynd::ndt::adapt_type::make(
          dynd::ndt::type::make<int64_t>(),
          dynd::ndt::datetime_type::make(dynd::tz_utc), "seconds since 1970");
    } else if (s == "ms") {
      dt = dynd::ndt::adapt_type::make(
          dynd::ndt::type::make<int64_t>(),
          dynd::ndt::datetime_type::make(dynd::tz_utc),
          "milliseconds since 1970");
    } else if (s == "us") {
      dt = dynd::ndt::adapt_type::make(
          dynd::ndt::type::make<int64_t>(),
          dynd::ndt::datetime_type::make(dynd::tz_utc),
          "microseconds since 1970");
    } else if (s == "ns") {
      dt = dynd::ndt::adapt_type::make(
          dynd::ndt::type::make<int64_t>(),
          dynd::ndt::datetime_type::make(dynd::tz_utc),
          "nanoseconds since 1970");
    }
    break;
  }
#endif // At least NumPy 1.6
  default:
    break;
  }

  if (dt.get_type_id() == dynd::uninitialized_type_id) {
    stringstream ss;
    ss << "unsupported Numpy dtype with type id " << d->type_num;
    throw dynd::type_error(ss.str());
  }

  if (!PyArray_ISNBO(d->byteorder)) {
    dt = dynd::ndt::byteswap_type::make(dt);
  }

  // If the data this dtype is for isn't aligned enough,
  // make an unaligned version.
  if (data_alignment != 0 && data_alignment < dt.get_data_alignment()) {
    dt = make_unaligned(dt);
  }

  return dt;
}

dynd::ndt::type pydynd::_type_from_numpy_type_num(int numpy_type_num)
{
  switch (numpy_type_num) {
  case NPY_BOOL:
    return dynd::ndt::type::make<dynd::bool1>();
  case NPY_BYTE:
    return dynd::ndt::type::make<npy_byte>();
  case NPY_UBYTE:
    return dynd::ndt::type::make<npy_ubyte>();
  case NPY_SHORT:
    return dynd::ndt::type::make<npy_short>();
  case NPY_USHORT:
    return dynd::ndt::type::make<npy_ushort>();
  case NPY_INT:
    return dynd::ndt::type::make<npy_int>();
  case NPY_UINT:
    return dynd::ndt::type::make<npy_uint>();
  case NPY_LONG:
    return dynd::ndt::type::make<npy_long>();
  case NPY_ULONG:
    return dynd::ndt::type::make<npy_ulong>();
  case NPY_LONGLONG:
    return dynd::ndt::type::make<npy_longlong>();
  case NPY_ULONGLONG:
    return dynd::ndt::type::make<npy_ulonglong>();
#if NPY_API_VERSION >= 6 // At least NumPy 1.6
  case NPY_HALF:
    return dynd::ndt::type::make<dynd::float16>();
#endif
  case NPY_FLOAT:
    return dynd::ndt::type::make<float>();
  case NPY_DOUBLE:
    return dynd::ndt::type::make<double>();
  case NPY_CFLOAT:
    return dynd::ndt::type::make<dynd::complex<float>>();
  case NPY_CDOUBLE:
    return dynd::ndt::type::make<dynd::complex<double>>();
  default: {
    stringstream ss;
    ss << "Cannot convert numpy type num " << numpy_type_num
       << " to a dynd type";
    throw dynd::type_error(ss.str());
  }
  }
}

void pydynd::fill_arrmeta_from_numpy_dtype(const dynd::ndt::type &dt,
                                           PyArray_Descr *d, char *arrmeta)
{
  switch (dt.get_type_id()) {
  case dynd::struct_type_id: {
    // In DyND, the struct offsets are part of the arrmeta instead of the dtype.
    // That's why we have to populate them here.
    PyObject *d_names = d->names;
    const dynd::ndt::struct_type *sdt = dt.extended<dynd::ndt::struct_type>();
    const uintptr_t *arrmeta_offsets = sdt->get_arrmeta_offsets_raw();
    size_t field_count = sdt->get_field_count();
    uintptr_t *offsets = reinterpret_cast<size_t *>(arrmeta);
    for (size_t i = 0; i < field_count; ++i) {
      PyObject *tup = PyDict_GetItem(d->fields, PyTuple_GET_ITEM(d_names, i));
      PyArray_Descr *fld_dtype;
      PyObject *title;
      int offset = 0;
      if (!PyArg_ParseTuple(tup, "Oi|O", &fld_dtype, &offset, &title)) {
        throw dynd::type_error("Numpy struct dtype has corrupt data");
      }
      // Set the field offset in the output arrmeta
      offsets[i] = offset;
      // Fill the arrmeta for the field, if necessary
      const dynd::ndt::type &ft = sdt->get_field_type(i);
      if (!ft.is_builtin()) {
        fill_arrmeta_from_numpy_dtype(ft, fld_dtype,
                                      arrmeta + arrmeta_offsets[i]);
      }
    }
    break;
  }
  case dynd::fixed_dim_type_id: {
    // The Numpy subarray becomes a series of fixed_dim types, so we
    // need to copy the strides into the arrmeta.
    dynd::ndt::type el;
    PyArray_ArrayDescr *adescr = d->subarray;
    if (adescr == NULL) {
      stringstream ss;
      ss << "Internal error building dynd arrmeta: Numpy dtype has "
            "NULL subarray corresponding to strided_dim type";
      throw dynd::type_error(ss.str());
    }
    if (PyTuple_Check(adescr->shape)) {
      int ndim = (int)PyTuple_GET_SIZE(adescr->shape);
      dynd::fixed_dim_type_arrmeta *md =
          reinterpret_cast<dynd::fixed_dim_type_arrmeta *>(arrmeta);
      intptr_t stride = adescr->base->elsize;
      el = dt;
      for (int i = ndim - 1; i >= 0; --i) {
        md[i].dim_size =
            pydynd::pyobject_as_index(PyTuple_GET_ITEM(adescr->shape, i));
        md[i].stride = stride;
        stride *= md[i].dim_size;
        el = el.extended<dynd::ndt::fixed_dim_type>()->get_element_type();
      }
      arrmeta += ndim * sizeof(dynd::fixed_dim_type_arrmeta);
    } else {
      dynd::fixed_dim_type_arrmeta *md =
          reinterpret_cast<dynd::fixed_dim_type_arrmeta *>(arrmeta);
      arrmeta += sizeof(dynd::fixed_dim_type_arrmeta);
      md->dim_size = pydynd::pyobject_as_index(adescr->shape);
      md->stride = adescr->base->elsize;
      el = dt.extended<dynd::ndt::fixed_dim_type>()->get_element_type();
    }
    // Fill the arrmeta for the array element, if necessary
    if (!el.is_builtin()) {
      fill_arrmeta_from_numpy_dtype(el, adescr->base, arrmeta);
    }
    break;
  }
  default:
    break;
  }
}

PyArray_Descr *pydynd::numpy_dtype_from__type(const dynd::ndt::type &tp)
{
  switch (tp.get_type_id()) {
  case dynd::bool_type_id:
    return PyArray_DescrFromType(NPY_BOOL);
  case dynd::int8_type_id:
    return PyArray_DescrFromType(NPY_INT8);
  case dynd::int16_type_id:
    return PyArray_DescrFromType(NPY_INT16);
  case dynd::int32_type_id:
    return PyArray_DescrFromType(NPY_INT32);
  case dynd::int64_type_id:
    return PyArray_DescrFromType(NPY_INT64);
  case dynd::uint8_type_id:
    return PyArray_DescrFromType(NPY_UINT8);
  case dynd::uint16_type_id:
    return PyArray_DescrFromType(NPY_UINT16);
  case dynd::uint32_type_id:
    return PyArray_DescrFromType(NPY_UINT32);
  case dynd::uint64_type_id:
    return PyArray_DescrFromType(NPY_UINT64);
  case dynd::float32_type_id:
    return PyArray_DescrFromType(NPY_FLOAT32);
  case dynd::float64_type_id:
    return PyArray_DescrFromType(NPY_FLOAT64);
  case dynd::complex_float32_type_id:
    return PyArray_DescrFromType(NPY_CFLOAT);
  case dynd::complex_float64_type_id:
    return PyArray_DescrFromType(NPY_CDOUBLE);
  case dynd::fixed_string_type_id: {
    const dynd::ndt::fixed_string_type *ftp =
        tp.extended<dynd::ndt::fixed_string_type>();
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
        case tuple_type_id: {
            const tuple_type *ttp = tp.extended<tuple_type>();
            const vector<ndt::type>& fields = ttp->get_fields();
            size_t num_fields = fields.size();
            const vector<size_t>& offsets = ttp->get_offsets();
            // TODO: Deal with the names better
            pyobject_ownref names_obj(PyList_New(num_fields));
            for (size_t i = 0; i < num_fields; ++i) {
                stringstream ss;
                ss << "f" << i;
                PyList_SET_ITEM((PyObject *)names_obj, i,
PyString_FromString(ss.str().c_str()));
            }
            pyobject_ownref formats_obj(PyList_New(num_fields));
            for (size_t i = 0; i < num_fields; ++i) {
                PyList_SET_ITEM((PyObject *)formats_obj, i, (PyObject
*)numpy_dtype_from__type(fields[i]));
            }
            pyobject_ownref offsets_obj(PyList_New(num_fields));
            for (size_t i = 0; i < num_fields; ++i) {
                PyList_SET_ITEM((PyObject *)offsets_obj, i,
PyLong_FromSize_t(offsets[i]));
            }
            pyobject_ownref itemsize_obj(PyLong_FromSize_t(tp.get_data_size()));
            pyobject_ownref dict_obj(PyDict_New());
            PyDict_SetItemString(dict_obj, "names", names_obj);
            PyDict_SetItemString(dict_obj, "formats", formats_obj);
            PyDict_SetItemString(dict_obj, "offsets", offsets_obj);
            PyDict_SetItemString(dict_obj, "itemsize", itemsize_obj);
            PyArray_Descr *result = NULL;
            if (PyArray_DescrConverter(dict_obj, &result) != NPY_SUCCEED) {
                throw dynd::type_error("failed to convert tuple dtype into numpy
dtype via dict");
            }
            return result;
        }
        case struct_type_id: {
            const struct_type *ttp = tp.extended<struct_type>();
            size_t field_count = ttp->get_field_count();
            size_t max_numpy_alignment = 1;
            std::vector<uintptr_t> offsets(field_count);
            struct_type::fill_default_data_offsets(field_count,
ttp->get_field_types_raw(), offsets.data());
            pyobject_ownref names_obj(PyList_New(field_count));
            for (size_t i = 0; i < field_count; ++i) {
                const string_type_data& fname = ttp->get_field_name_raw(i);
#if PY_VERSION_HEX >= 0x03000000
                pyobject_ownref name_str(PyUnicode_FromStringAndSize(
                    fname.begin, fname.end - fname.begin));
#else
                pyobject_ownref name_str(PyString_FromStringAndSize(
                    fname.begin, fname.end - fname.begin));
#endif
                PyList_SET_ITEM((PyObject *)names_obj, i, name_str.release());
            }
            pyobject_ownref formats_obj(PyList_New(field_count));
            for (size_t i = 0; i < field_count; ++i) {
                PyArray_Descr *npdt =
numpy_dtype_from__type(ttp->get_field_type(i));
                max_numpy_alignment = max(max_numpy_alignment,
(size_t)npdt->alignment);
                PyList_SET_ITEM((PyObject *)formats_obj, i, (PyObject *)npdt);
            }
            pyobject_ownref offsets_obj(PyList_New(field_count));
            for (size_t i = 0; i < field_count; ++i) {
                PyList_SET_ITEM((PyObject *)offsets_obj, i,
PyLong_FromSize_t(offsets[i]));
            }
            pyobject_ownref
itemsize_obj(PyLong_FromSize_t(tp.get_default_data_size()));
            pyobject_ownref dict_obj(PyDict_New());
            PyDict_SetItemString(dict_obj, "names", names_obj);
            PyDict_SetItemString(dict_obj, "formats", formats_obj);
            PyDict_SetItemString(dict_obj, "offsets", offsets_obj);
            PyDict_SetItemString(dict_obj, "itemsize", itemsize_obj);
            // This isn't quite right, but the rules between numpy and dynd
            // differ enough to make this tricky.
            if (max_numpy_alignment > 1 &&
                            max_numpy_alignment == tp.get_data_alignment()) {
                Py_INCREF(Py_True);
                PyDict_SetItemString(dict_obj, "aligned", Py_True);
            }
            PyArray_Descr *result = NULL;
            if (PyArray_DescrConverter(dict_obj, &result) != NPY_SUCCEED) {
                stringstream ss;
                ss << "failed to convert dtype " << tp << " into numpy dtype via
dict";
                throw dynd::type_error(ss.str());
            }
            return result;
        }
        case fixed_dim_type_id: {
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
            } while (child_tp.get_type_id() == cfixed_dim_type_id);
            pyobject_ownref dtype_obj((PyObject
*)numpy_dtype_from__type(child_tp));
            pyobject_ownref shape_obj(intptr_array_as_tuple((int)shape.size(),
&shape[0]));
            pyobject_ownref tuple_obj(PyTuple_New(2));
            PyTuple_SET_ITEM(tuple_obj.get(), 0, dtype_obj.release());
            PyTuple_SET_ITEM(tuple_obj.get(), 1, shape_obj.release());
            PyArray_Descr *result = NULL;
            if (PyArray_DescrConverter(tuple_obj, &result) != NPY_SUCCEED) {
                throw dynd::type_error("failed to convert dynd type into numpy
subarray dtype");
            }
            return result;
        }
*/
  case dynd::view_type_id: {
    // If there's a view which is for alignment purposes, throw it
    // away because Numpy works differently
    if (tp.operand_type().get_type_id() == dynd::fixed_bytes_type_id) {
      return numpy_dtype_from__type(tp.value_type());
    }
    break;
  }
  case dynd::byteswap_type_id: {
    // If it's a simple byteswap from bytes, that can be converted
    if (tp.operand_type().get_type_id() == dynd::fixed_bytes_type_id) {
      PyArray_Descr *unswapped = numpy_dtype_from__type(tp.value_type());
      PyArray_Descr *result = PyArray_DescrNewByteorder(unswapped, NPY_SWAP);
      Py_DECREF(unswapped);
      return result;
    }
  }
  default:
    break;
  }

  stringstream ss;
  ss << "cannot convert dynd type " << tp << " into a Numpy dtype";
  throw dynd::type_error(ss.str());
}

PyArray_Descr *pydynd::numpy_dtype_from__type(const dynd::ndt::type &tp,
                                              const char *arrmeta)
{
  switch (tp.get_type_id()) {
  case dynd::struct_type_id: {
    throw std::runtime_error("converting");
    if (arrmeta == NULL) {
      stringstream ss;
      ss << "Can only convert dynd type " << tp
         << " into a numpy dtype with array arrmeta";
      throw dynd::type_error(ss.str());
    }
    const dynd::ndt::struct_type *stp = tp.extended<dynd::ndt::struct_type>();
    const uintptr_t *arrmeta_offsets = stp->get_arrmeta_offsets_raw();
    const uintptr_t *offsets = stp->get_data_offsets(arrmeta);
    size_t field_count = stp->get_field_count();
    size_t max_numpy_alignment = 1;

    pyobject_ownref names_obj(PyList_New(field_count));
    for (size_t i = 0; i < field_count; ++i) {
      const dynd::string &fname = stp->get_field_name_raw(i);
#if PY_VERSION_HEX >= 0x03000000
      pyobject_ownref name_str(PyUnicode_FromStringAndSize(
          fname.begin(), fname.end() - fname.begin()));
#else
      pyobject_ownref name_str(PyString_FromStringAndSize(
          fname.begin(), fname.end() - fname.begin()));
#endif
      PyList_SET_ITEM((PyObject *)names_obj, i, name_str.release());
    }

    pyobject_ownref formats_obj(PyList_New(field_count));
    for (size_t i = 0; i < field_count; ++i) {
      PyArray_Descr *npdt = numpy_dtype_from__type(
          stp->get_field_type(i), arrmeta + arrmeta_offsets[i]);
      max_numpy_alignment = max(max_numpy_alignment, (size_t)npdt->alignment);
      PyList_SET_ITEM((PyObject *)formats_obj, i, (PyObject *)npdt);
    }

    pyobject_ownref offsets_obj(PyList_New(field_count));
    for (size_t i = 0; i < field_count; ++i) {
      PyList_SET_ITEM((PyObject *)offsets_obj, i,
                      PyLong_FromSize_t(offsets[i]));
    }

    pyobject_ownref itemsize_obj(PyLong_FromSize_t(tp.get_data_size()));

    pyobject_ownref dict_obj(PyDict_New());
    PyDict_SetItemString(dict_obj, "names", names_obj);
    PyDict_SetItemString(dict_obj, "formats", formats_obj);
    PyDict_SetItemString(dict_obj, "offsets", offsets_obj);
    PyDict_SetItemString(dict_obj, "itemsize", itemsize_obj);
    // This isn't quite right, but the rules between numpy and dynd
    // differ enough to make this tricky.
    if (max_numpy_alignment > 1 &&
        max_numpy_alignment == tp.get_data_alignment()) {
      Py_INCREF(Py_True);
      PyDict_SetItemString(dict_obj, "aligned", Py_True);
    }

    PyArray_Descr *result = NULL;
    if (PyArray_DescrConverter(dict_obj, &result) != NPY_SUCCEED) {
      throw dynd::type_error(
          "failed to convert dtype into numpy struct dtype via dict");
    }
    return result;
  }
  default:
    return numpy_dtype_from__type(tp);
  }
}

int pydynd::_type_from_numpy_scalar_typeobject(PyTypeObject *obj,
                                               dynd::ndt::type &out_d)
{
  if (obj == &PyBoolArrType_Type) {
    out_d = dynd::ndt::type::make<dynd::bool1>();
  } else if (obj == &PyByteArrType_Type) {
    out_d = dynd::ndt::type::make<npy_byte>();
  } else if (obj == &PyUByteArrType_Type) {
    out_d = dynd::ndt::type::make<npy_ubyte>();
  } else if (obj == &PyShortArrType_Type) {
    out_d = dynd::ndt::type::make<npy_short>();
  } else if (obj == &PyUShortArrType_Type) {
    out_d = dynd::ndt::type::make<npy_ushort>();
  } else if (obj == &PyIntArrType_Type) {
    out_d = dynd::ndt::type::make<npy_int>();
  } else if (obj == &PyUIntArrType_Type) {
    out_d = dynd::ndt::type::make<npy_uint>();
  } else if (obj == &PyLongArrType_Type) {
    out_d = dynd::ndt::type::make<npy_long>();
  } else if (obj == &PyULongArrType_Type) {
    out_d = dynd::ndt::type::make<npy_ulong>();
  } else if (obj == &PyLongLongArrType_Type) {
    out_d = dynd::ndt::type::make<npy_longlong>();
  } else if (obj == &PyULongLongArrType_Type) {
    out_d = dynd::ndt::type::make<npy_ulonglong>();
  } else if (obj == &PyFloatArrType_Type) {
    out_d = dynd::ndt::type::make<npy_float>();
  } else if (obj == &PyDoubleArrType_Type) {
    out_d = dynd::ndt::type::make<npy_double>();
  } else if (obj == &PyCFloatArrType_Type) {
    out_d = dynd::ndt::type::make<dynd::complex<float>>();
  } else if (obj == &PyCDoubleArrType_Type) {
    out_d = dynd::ndt::type::make<dynd::complex<double>>();
  } else {
    return -1;
  }

  return 0;
}

dynd::ndt::type pydynd::_type_of_numpy_scalar(PyObject *obj)
{
  if (PyArray_IsScalar(obj, Bool)) {
    return dynd::ndt::type::make<dynd::bool1>();
  } else if (PyArray_IsScalar(obj, Byte)) {
    return dynd::ndt::type::make<npy_byte>();
  } else if (PyArray_IsScalar(obj, UByte)) {
    return dynd::ndt::type::make<npy_ubyte>();
  } else if (PyArray_IsScalar(obj, Short)) {
    return dynd::ndt::type::make<npy_short>();
  } else if (PyArray_IsScalar(obj, UShort)) {
    return dynd::ndt::type::make<npy_ushort>();
  } else if (PyArray_IsScalar(obj, Int)) {
    return dynd::ndt::type::make<npy_int>();
  } else if (PyArray_IsScalar(obj, UInt)) {
    return dynd::ndt::type::make<npy_uint>();
  } else if (PyArray_IsScalar(obj, Long)) {
    return dynd::ndt::type::make<npy_long>();
  } else if (PyArray_IsScalar(obj, ULong)) {
    return dynd::ndt::type::make<npy_ulong>();
  } else if (PyArray_IsScalar(obj, LongLong)) {
    return dynd::ndt::type::make<npy_longlong>();
  } else if (PyArray_IsScalar(obj, ULongLong)) {
    return dynd::ndt::type::make<npy_ulonglong>();
  } else if (PyArray_IsScalar(obj, Float)) {
    return dynd::ndt::type::make<float>();
  } else if (PyArray_IsScalar(obj, Double)) {
    return dynd::ndt::type::make<double>();
  } else if (PyArray_IsScalar(obj, CFloat)) {
    return dynd::ndt::type::make<dynd::complex<float>>();
  } else if (PyArray_IsScalar(obj, CDouble)) {
    return dynd::ndt::type::make<dynd::complex<double>>();
  }

  throw dynd::type_error(
      "could not deduce a pydynd type from the numpy scalar object");
}

inline size_t get_alignment_of(uintptr_t align_bits)
{
  size_t alignment = 1;
  // Loop 4 times, maximum alignment of 16
  for (int i = 0; i < 4; ++i) {
    if ((align_bits & alignment) == 0) {
      alignment <<= 1;
    } else {
      return alignment;
    }
  }
  return alignment;
}

inline size_t get_alignment_of(PyArrayObject *obj)
{
  // Get the alignment of the data
  uintptr_t align_bits = reinterpret_cast<uintptr_t>(PyArray_DATA(obj));
  int ndim = PyArray_NDIM(obj);
  intptr_t *strides = PyArray_STRIDES(obj);
  for (int idim = 0; idim < ndim; ++idim) {
    align_bits |= (uintptr_t)strides[idim];
  }

  return get_alignment_of(align_bits);
}

dynd::nd::array pydynd::array_from_numpy_array(PyArrayObject *obj,
                                               uint32_t access_flags,
                                               bool always_copy)
{
  // If a copy isn't requested, make sure the access flags are ok
  if (!always_copy) {
    if ((access_flags & dynd::nd::write_access_flag) &&
        !PyArray_ISWRITEABLE(obj)) {
      throw runtime_error("cannot view a readonly numpy array as readwrite");
    }
    if (access_flags & dynd::nd::immutable_access_flag) {
      throw runtime_error("cannot view a numpy array as immutable");
    }
  }

  PyArray_Descr *dtype = PyArray_DESCR(obj);

  if (always_copy || PyDataType_FLAGCHK(dtype, NPY_ITEM_HASOBJECT)) {
    // TODO would be nicer without the extra type transformation of the
    // get_canonical_type call
    dynd::nd::array result =
        dynd::nd::dtyped_empty(PyArray_NDIM(obj), PyArray_SHAPE(obj),
                               pydynd::_type_from_numpy_dtype(
                                   PyArray_DESCR(obj)).get_canonical_type());
    pydynd::nd::array_copy_from_numpy(result.get_type(), result.get()->metadata(),
                                      result.data(), obj,
                                      &dynd::eval::default_eval_context);
    if (access_flags != 0) {
      // Use the requested access flags
      result.get()->flags = access_flags;
    } else {
      result.get()->flags = dynd::nd::default_access_flags;
    }
    return result;
  } else {
    // Get the dtype of the array
    dynd::ndt::type d = pydynd::_type_from_numpy_dtype(PyArray_DESCR(obj),
                                                       get_alignment_of(obj));

    // Get a shared pointer that tracks buffer ownership
    PyObject *base = PyArray_BASE(obj);
    dynd::intrusive_ptr<dynd::memory_block_data> memblock;
    if (base == NULL || (PyArray_FLAGS(obj) & NPY_ARRAY_UPDATEIFCOPY) != 0) {
      Py_INCREF(obj);
      memblock = dynd::make_external_memory_block(obj, py_decref_function);
    } else {
      if (DyND_PyArray_CheckExact(base)) {
        // If the base of the numpy array is an nd::array, skip the Python
        // reference
        memblock = ((DyND_PyArrayObject *)base)->v.get_data_memblock();
      } else {
        Py_INCREF(base);
        memblock = dynd::make_external_memory_block(base, py_decref_function);
      }
    }

    // Create the result nd::array
    char *arrmeta = NULL;
    dynd::nd::array result = dynd::nd::make_strided_array_from_data(
        d, PyArray_NDIM(obj), PyArray_DIMS(obj), PyArray_STRIDES(obj),
        dynd::nd::read_access_flag |
            (PyArray_ISWRITEABLE(obj) ? dynd::nd::write_access_flag : 0),
        PyArray_BYTES(obj), std::move(memblock), &arrmeta);
    if (d.get_type_id() == dynd::struct_type_id) {
      // If it's a struct, there's additional arrmeta that needs to be populated
      pydynd::fill_arrmeta_from_numpy_dtype(d, PyArray_DESCR(obj), arrmeta);
    }

    if (access_flags != 0) {
      // Use the requested access flags
      result.get()->flags = access_flags;
    }
    return result;
  }
}

dynd::nd::array pydynd::array_from_numpy_scalar(PyObject *obj,
                                                uint32_t access_flags)
{
  dynd::nd::array result;
  if (PyArray_IsScalar(obj, Bool)) {
    result =
        dynd::nd::array((dynd::bool1)(((PyBoolScalarObject *)obj)->obval != 0));
  } else if (PyArray_IsScalar(obj, Byte)) {
    result = dynd::nd::array(((PyByteScalarObject *)obj)->obval);
  } else if (PyArray_IsScalar(obj, UByte)) {
    result = dynd::nd::array(((PyUByteScalarObject *)obj)->obval);
  } else if (PyArray_IsScalar(obj, Short)) {
    result = dynd::nd::array(((PyShortScalarObject *)obj)->obval);
  } else if (PyArray_IsScalar(obj, UShort)) {
    result = dynd::nd::array(((PyUShortScalarObject *)obj)->obval);
  } else if (PyArray_IsScalar(obj, Int)) {
    result = dynd::nd::array(((PyIntScalarObject *)obj)->obval);
  } else if (PyArray_IsScalar(obj, UInt)) {
    result = dynd::nd::array(((PyUIntScalarObject *)obj)->obval);
  } else if (PyArray_IsScalar(obj, Long)) {
    result = dynd::nd::array(((PyLongScalarObject *)obj)->obval);
  } else if (PyArray_IsScalar(obj, ULong)) {
    result = dynd::nd::array(((PyULongScalarObject *)obj)->obval);
  } else if (PyArray_IsScalar(obj, LongLong)) {
    result = dynd::nd::array(((PyLongLongScalarObject *)obj)->obval);
  } else if (PyArray_IsScalar(obj, ULongLong)) {
    result = dynd::nd::array(((PyULongLongScalarObject *)obj)->obval);
  } else if (PyArray_IsScalar(obj, Float)) {
    result = dynd::nd::array(((PyFloatScalarObject *)obj)->obval);
  } else if (PyArray_IsScalar(obj, Double)) {
    result = dynd::nd::array(((PyDoubleScalarObject *)obj)->obval);
  } else if (PyArray_IsScalar(obj, CFloat)) {
    npy_cfloat &val = ((PyCFloatScalarObject *)obj)->obval;
    result = dynd::nd::array(dynd::complex<float>(val.real, val.imag));
  } else if (PyArray_IsScalar(obj, CDouble)) {
    npy_cdouble &val = ((PyCDoubleScalarObject *)obj)->obval;
    result = dynd::nd::array(dynd::complex<double>(val.real, val.imag));
#if NPY_API_VERSION >= 6 // At least NumPy 1.6
  } else if (PyArray_IsScalar(obj, Datetime)) {
    const PyDatetimeScalarObject *scalar = (PyDatetimeScalarObject *)obj;
    int64_t val = scalar->obval;
    if (scalar->obmeta.base <= NPY_FR_D) {
      result = dynd::nd::empty(dynd::ndt::date_type::make());
      int32_t result_val;
      if (val == NPY_DATETIME_NAT) {
        result_val = DYND_DATE_NA;
      } else {
        dynd::date_ymd ymd;
        switch (scalar->obmeta.base) {
        case NPY_FR_Y:
          ymd.year = static_cast<int16_t>(val + 1970);
          ymd.month = 1;
          ymd.day = 1;
          result_val = ymd.to_days();
          break;
        case NPY_FR_M:
          if (val >= 0) {
            ymd.year = static_cast<int16_t>(val / 12 + 1970);
          } else {
            ymd.year = static_cast<int16_t>((val - 11) / 12 + 1970);
          }
          ymd.month = static_cast<int8_t>(val - (ymd.year - 1970) * 12 + 1);
          ymd.day = 1;
          result_val = ymd.to_days();
          break;
        case NPY_FR_D:
          result_val = static_cast<int32_t>(val);
          break;
        default:
          throw dynd::type_error("Unsupported NumPy date unit");
        }
      }
      *reinterpret_cast<int32_t *>(result.data()) = result_val;
    } else {
      result = dynd::nd::empty(dynd::ndt::datetime_type::make(dynd::tz_utc));
      int64_t result_val;
      switch (scalar->obmeta.base) {
      case NPY_FR_h:
        result_val = val * DYND_TICKS_PER_HOUR;
        break;
      case NPY_FR_m:
        result_val = val * DYND_TICKS_PER_MINUTE;
        break;
      case NPY_FR_s:
        result_val = val * DYND_TICKS_PER_SECOND;
        break;
      case NPY_FR_ms:
        result_val = val * DYND_TICKS_PER_MILLISECOND;
        break;
      case NPY_FR_us:
        result_val = val * DYND_TICKS_PER_MICROSECOND;
        break;
      case NPY_FR_ns:
        if (val >= 0) {
          result_val = val / DYND_NANOSECONDS_PER_TICK;
        } else {
          result_val =
              (val - DYND_NANOSECONDS_PER_TICK + 1) / DYND_NANOSECONDS_PER_TICK;
        }
        break;
      default:
        throw dynd::type_error("Unsupported NumPy datetime unit");
      }
      *reinterpret_cast<int64_t *>(result.data()) = result_val;
    }
#endif
  } else if (PyArray_IsScalar(obj, Void)) {
    pyobject_ownref arr(PyArray_FromAny(obj, NULL, 0, 0, 0, NULL));
    return array_from_numpy_array((PyArrayObject *)arr.get(), access_flags,
                                  true);
  } else {
    stringstream ss;
    pyobject_ownref obj_tp(PyObject_Repr((PyObject *)Py_TYPE(obj)));
    ss << "could not create a dynd array from the numpy scalar object";
    ss << " of type " << pydynd::pystring_as_string(obj_tp.get());
    throw dynd::type_error(ss.str());
  }

  result.get()->flags =
      access_flags ? access_flags : dynd::nd::default_access_flags;

  return result;
}

char pydynd::numpy_kindchar_of(const dynd::ndt::type &d)
{
  switch (d.get_kind()) {
  case dynd::bool_kind:
    return 'b';
  case dynd::sint_kind:
    return 'i';
  case dynd::uint_kind:
    return 'u';
  case dynd::real_kind:
    return 'f';
  case dynd::complex_kind:
    return 'c';
  case dynd::string_kind:
    if (d.get_type_id() == dynd::fixed_string_type_id) {
      const dynd::ndt::base_string_type *esd =
          d.extended<dynd::ndt::base_string_type>();
      switch (esd->get_encoding()) {
      case dynd::string_encoding_ascii:
        return 'S';
      case dynd::string_encoding_utf_32:
        return 'U';
      default:
        break;
      }
    }
    break;
  default:
    break;
  }

  stringstream ss;
  ss << "dynd type \"" << d << "\" does not have an equivalent numpy kind";
  throw dynd::type_error(ss.str());
}

#endif // DYND_NUMPY_INTEROP

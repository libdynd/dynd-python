//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>

#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/pointer_type.hpp>
#include <dynd/types/type_alignment.hpp>

#include "ctypes_interop.hpp"
#include "type_functions.hpp"
#include "utility_functions.hpp"

using namespace std;
using namespace dynd;
using namespace pydynd;

ctypes_info pydynd::ctypes;

void pydynd::init_ctypes_interop()
{
  memset(&ctypes, 0, sizeof(ctypes));

  // The C _ctypes module
  ctypes._ctypes = PyImport_ImportModule("_ctypes");
  if (ctypes._ctypes == NULL) {
    throw runtime_error("Could not import module _ctypes");
  }

  // The internal type objects used by ctypes
  ctypes.PyCStructType_Type =
      PyObject_GetAttrString(ctypes._ctypes, "Structure");
  // _ctypes doesn't expose PyCData_Type, but we know it's the base class of
  // PyCStructType_Type
  ctypes.PyCData_Type =
      (PyObject *)((PyTypeObject *)ctypes.PyCStructType_Type)->tp_base;
  ctypes.UnionType_Type = PyObject_GetAttrString(ctypes._ctypes, "Union");
  ctypes.PyCPointerType_Type =
      PyObject_GetAttrString(ctypes._ctypes, "_Pointer");
  ctypes.PyCArrayType_Type = PyObject_GetAttrString(ctypes._ctypes, "Array");
  ctypes.PyCSimpleType_Type =
      PyObject_GetAttrString(ctypes._ctypes, "_SimpleCData");
  ctypes.PyCFuncPtrType_Type =
      PyObject_GetAttrString(ctypes._ctypes, "CFuncPtr");

  if (PyErr_Occurred()) {
    Py_XDECREF(ctypes._ctypes);

    Py_XDECREF(ctypes.PyCData_Type);
    Py_XDECREF(ctypes.PyCStructType_Type);
    Py_XDECREF(ctypes.UnionType_Type);
    Py_XDECREF(ctypes.PyCPointerType_Type);
    Py_XDECREF(ctypes.PyCArrayType_Type);
    Py_XDECREF(ctypes.PyCSimpleType_Type);
    Py_XDECREF(ctypes.PyCFuncPtrType_Type);

    memset(&ctypes, 0, sizeof(ctypes));
    throw std::runtime_error(
        "Error initializing ctypes C-level data for low level interop");
  }
}

void pydynd::get_ctypes_signature(PyCFuncPtrObject *cfunc,
                                  ndt::type &out_returntype,
                                  std::vector<dynd::ndt::type> &out_paramtypes)
{
  // The fields restype and argtypes are not always stored at the C level,
  // so must use the higher level getattr.
  pyobject_ownref restype(PyObject_GetAttrString((PyObject *)cfunc, "restype"));
  pyobject_ownref argtypes(
      PyObject_GetAttrString((PyObject *)cfunc, "argtypes"));

  if (argtypes == Py_None) {
    throw std::runtime_error("The argtypes and restype of a ctypes function "
                             "pointer must be specified to get its signature");
  }

  // Get the return type
  if (restype == Py_None) {
    // No return type
    out_returntype = ndt::type::make<void>();
  }
  else {
    out_returntype = _type_from_ctypes_cdatatype(restype);
  }

  Py_ssize_t argcount = PySequence_Size(argtypes);
  if (argcount < 0) {
    throw runtime_error(
        "The argtypes of the ctypes function pointer has the wrong type");
  }

  // Set the output size
  out_paramtypes.resize(argcount);

  // Get the argument types
  for (intptr_t i = 0; i < argcount; ++i) {
    pyobject_ownref element(PySequence_GetItem(argtypes, i));
    out_paramtypes[i] = _type_from_ctypes_cdatatype(element);
  }
}

dynd::ndt::type pydynd::_type_from_ctypes_cdatatype(PyObject *d)
{
  if (!PyObject_IsSubclass(d, ctypes.PyCData_Type)) {
    throw runtime_error("internal error: requested a dynd type from a ctypes c "
                        "data type, but the given object has the wrong type");
  }

  // If the ctypes type has a _dynd_type_ property, that should be
  // a pydynd type instance corresponding to the type. This is how
  // the complex type is supported, for example.
  PyObject *dynd_type_obj = PyObject_GetAttrString(d, "_dynd_type_");
  if (dynd_type_obj == NULL) {
    PyErr_Clear();
  }
  else {
    pyobject_ownref dynd_type(dynd_type_obj);
    return make__type_from_pyobject(dynd_type);
  }

  // The simple C data types
  if (PyObject_IsSubclass(d, ctypes.PyCSimpleType_Type)) {
    pyobject_ownref proto(PyObject_GetAttrString(d, "_type_"));
    std::string proto_str = pystring_as_string(proto);
    if (proto_str.size() != 1) {
      throw std::runtime_error(
          "invalid ctypes type, its _type_ value is incorrect");
    }

    switch (proto_str[0]) {
    case 'b':
      return ndt::type::make<int8_t>();
    case 'B':
      return ndt::type::make<uint8_t>();
    case 'c':
      return ndt::fixed_string_type::make(1, string_encoding_ascii);
    case 'd':
      return ndt::type::make<double>();
    case 'f':
      return ndt::type::make<float>();
    case 'h':
      return ndt::type::make<int16_t>();
    case 'H':
      return ndt::type::make<uint16_t>();
    case 'i':
      return ndt::type::make<int32_t>();
    case 'I':
      return ndt::type::make<uint32_t>();
    case 'l':
      return ndt::type::make<long>();
    case 'L':
      return ndt::type::make<unsigned long>();
    case 'q':
      return ndt::type::make<int64_t>();
    case 'Q':
      return ndt::type::make<uint64_t>();
    default: {
      stringstream ss;
      ss << "The ctypes type code '" << proto_str[0]
         << "' cannot be converted to a dynd type";
      throw runtime_error(ss.str());
    }
    }
  }
  else if (PyObject_IsSubclass(d, ctypes.PyCPointerType_Type)) {
    // Translate into a blockref pointer type
    pyobject_ownref target_tp_obj(PyObject_GetAttrString(d, "_type_"));
    ndt::type target_tp = _type_from_ctypes_cdatatype(target_tp_obj);
    return ndt::pointer_type::make(target_tp);
  }
  else if (PyObject_IsSubclass(d, ctypes.PyCStructType_Type)) {
    // Translate into a struct type
    pyobject_ownref fields_list_obj(PyObject_GetAttrString(d, "_fields_"));
    if (!PyList_Check(fields_list_obj.get())) {
      throw runtime_error(
          "The _fields_ member of the ctypes C struct is not a list");
    }
    vector<ndt::type> field_types;
    vector<std::string> field_names;
    vector<size_t> field_offsets;
    Py_ssize_t field_count = PyList_GET_SIZE(fields_list_obj.get());
    for (Py_ssize_t i = 0; i < field_count; ++i) {
      PyObject *item = PyList_GET_ITEM(fields_list_obj.get(), i);
      if (!PyTuple_Check(item) || PyTuple_GET_SIZE(item) != 2) {
        stringstream ss;
        ss << "The _fields_[" << i
           << "] member of the ctypes C struct is not a tuple of size 2";
        throw runtime_error(ss.str());
      }
      field_types.push_back(
          _type_from_ctypes_cdatatype(PyTuple_GET_ITEM(item, 1)));
      PyObject *key = PyTuple_GET_ITEM(item, 0);
      field_names.push_back(pystring_as_string(key));
      pyobject_ownref field_data_obj(PyObject_GetAttr(d, key));
      pyobject_ownref field_data_offset_obj(
          PyObject_GetAttrString(field_data_obj.get(), "offset"));
      field_offsets.push_back(pyobject_as_index(field_data_offset_obj.get()));
      // If the field isn't aligned as the type requires, make it into an
      // unaligned version
      if (!offset_is_aligned(field_offsets.back(),
                             field_types.back().get_data_alignment())) {
        field_types.back() = make_unaligned(field_types.back());
      }
    }
    pyobject_ownref total_size_obj(
        PyObject_CallMethod(ctypes._ctypes, (char *)"sizeof", (char *)"N", d));
    size_t total_size = pyobject_as_index(total_size_obj.get());

    return ndt::struct_type::make(field_names, field_types);
  }
  else if (PyObject_IsSubclass(d, ctypes.PyCArrayType_Type)) {
    // Translate into a fixed_dim
    pyobject_ownref array_length_obj(PyObject_GetAttrString(d, "_length_"));
    intptr_t array_length = pyobject_as_index(array_length_obj.get());
    pyobject_ownref element_tp_obj(PyObject_GetAttrString(d, "_type_"));
    ndt::type element_tp = _type_from_ctypes_cdatatype(element_tp_obj);
    return ndt::make_fixed_dim(array_length, element_tp);
  }

  throw runtime_error("Ctypes type object is not supported by dynd type");
}

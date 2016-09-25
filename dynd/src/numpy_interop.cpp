//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "numpy_interop.hpp"

#if DYND_NUMPY_INTEROP

#include <dynd/memblock/external_memory_block.hpp>

#include "array_functions.hpp"
#include "copy_from_numpy_arrfunc.hpp"

#include <numpy/arrayscalars.h>

using namespace std;

void pydynd::fill_arrmeta_from_numpy_dtype(const dynd::ndt::type &dt, PyArray_Descr *d, char *arrmeta)
{
  switch (dt.get_id()) {
  case dynd::struct_id: {
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
        fill_arrmeta_from_numpy_dtype(ft, fld_dtype, arrmeta + arrmeta_offsets[i]);
      }
    }
    break;
  }
  case dynd::fixed_dim_id: {
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
      dynd::fixed_dim_type_arrmeta *md = reinterpret_cast<dynd::fixed_dim_type_arrmeta *>(arrmeta);
      intptr_t stride = adescr->base->elsize;
      el = dt;
      for (int i = ndim - 1; i >= 0; --i) {
        md[i].dim_size = pydynd::pyobject_as_index(PyTuple_GET_ITEM(adescr->shape, i));
        md[i].stride = stride;
        stride *= md[i].dim_size;
        el = el.extended<dynd::ndt::fixed_dim_type>()->get_element_type();
      }
      arrmeta += ndim * sizeof(dynd::fixed_dim_type_arrmeta);
    }
    else {
      dynd::fixed_dim_type_arrmeta *md = reinterpret_cast<dynd::fixed_dim_type_arrmeta *>(arrmeta);
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

dynd::nd::array pydynd::array_from_numpy_array(PyArrayObject *obj, uint32_t access_flags, bool always_copy)
{
  // If a copy isn't requested, make sure the access flags are ok
  if (!always_copy) {
    if ((access_flags & dynd::nd::write_access_flag) && !PyArray_ISWRITEABLE(obj)) {
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
    dynd::nd::array result = dynd::nd::dtyped_empty(
        PyArray_NDIM(obj), PyArray_SHAPE(obj), pydynd::_type_from_numpy_dtype(PyArray_DESCR(obj)).get_canonical_type());
    pydynd::nd::array_copy_from_numpy(result.get_type(), result.get()->metadata(), result.data(), obj,
                                      &dynd::eval::default_eval_context);
    return result;
  }
  else {
    // Get the dtype of the array
    dynd::ndt::type d = pydynd::_type_from_numpy_dtype(PyArray_DESCR(obj), get_alignment_of(obj));

    // Get a shared pointer that tracks buffer ownership
    PyObject *base = PyArray_BASE(obj);
    dynd::nd::memory_block memblock;
    if (base == NULL || (PyArray_FLAGS(obj) & NPY_ARRAY_UPDATEIFCOPY) != 0) {
      Py_INCREF(obj);
      memblock = dynd::nd::make_memory_block<dynd::nd::external_memory_block>(obj, py_decref_function);
    }
    else {
      if (PyObject_TypeCheck(base, pydynd::get_array_pytypeobject())) {
        // If the base of the numpy array is an nd::array, skip the Python
        // reference
        memblock = pydynd::array_to_cpp_ref(base).get_data_memblock();
      }
      else {
        Py_INCREF(base);
        memblock = dynd::nd::make_memory_block<dynd::nd::external_memory_block>(base, py_decref_function);
      }
    }

    // Create the result nd::array
    char *arrmeta = NULL;
    dynd::nd::array result = dynd::nd::make_strided_array_from_data(
        d, PyArray_NDIM(obj), PyArray_DIMS(obj), PyArray_STRIDES(obj),
        dynd::nd::read_access_flag | (PyArray_ISWRITEABLE(obj) ? dynd::nd::write_access_flag : 0), PyArray_BYTES(obj),
        dynd::nd::memory_block(std::move(memblock).get(), true), &arrmeta);
    if (d.get_id() == dynd::struct_id) {
      // If it's a struct, there's additional arrmeta that needs to be populated
      pydynd::fill_arrmeta_from_numpy_dtype(d, PyArray_DESCR(obj), arrmeta);
    }

    return result;
  }
}

dynd::nd::array pydynd::array_from_numpy_scalar(PyObject *obj, uint32_t access_flags)
{
  dynd::nd::array result;
  if (PyArray_IsScalar(obj, Bool)) {
    result = dynd::nd::array((dynd::bool1)(((PyBoolScalarObject *)obj)->obval != 0));
  }
  else if (PyArray_IsScalar(obj, Byte)) {
    result = dynd::nd::array(((PyByteScalarObject *)obj)->obval);
  }
  else if (PyArray_IsScalar(obj, UByte)) {
    result = dynd::nd::array(((PyUByteScalarObject *)obj)->obval);
  }
  else if (PyArray_IsScalar(obj, Short)) {
    result = dynd::nd::array(((PyShortScalarObject *)obj)->obval);
  }
  else if (PyArray_IsScalar(obj, UShort)) {
    result = dynd::nd::array(((PyUShortScalarObject *)obj)->obval);
  }
  else if (PyArray_IsScalar(obj, Int)) {
    result = dynd::nd::array(((PyIntScalarObject *)obj)->obval);
  }
  else if (PyArray_IsScalar(obj, UInt)) {
    result = dynd::nd::array(((PyUIntScalarObject *)obj)->obval);
  }
  else if (PyArray_IsScalar(obj, Long)) {
    result = dynd::nd::array(((PyLongScalarObject *)obj)->obval);
  }
  else if (PyArray_IsScalar(obj, ULong)) {
    result = dynd::nd::array(((PyULongScalarObject *)obj)->obval);
  }
  else if (PyArray_IsScalar(obj, LongLong)) {
    result = dynd::nd::array(((PyLongLongScalarObject *)obj)->obval);
  }
  else if (PyArray_IsScalar(obj, ULongLong)) {
    result = dynd::nd::array(((PyULongLongScalarObject *)obj)->obval);
  }
  else if (PyArray_IsScalar(obj, Float)) {
    result = dynd::nd::array(((PyFloatScalarObject *)obj)->obval);
  }
  else if (PyArray_IsScalar(obj, Double)) {
    result = dynd::nd::array(((PyDoubleScalarObject *)obj)->obval);
  }
  else if (PyArray_IsScalar(obj, CFloat)) {
    npy_cfloat &val = ((PyCFloatScalarObject *)obj)->obval;
    result = dynd::nd::array(dynd::complex<float>(val.real, val.imag));
  }
  else if (PyArray_IsScalar(obj, CDouble)) {
    npy_cdouble &val = ((PyCDoubleScalarObject *)obj)->obval;
    result = dynd::nd::array(dynd::complex<double>(val.real, val.imag));
  }
  else if (PyArray_IsScalar(obj, Void)) {
    py_ref arr = capture_if_not_null(PyArray_FromAny(obj, NULL, 0, 0, 0, NULL));
    return array_from_numpy_array((PyArrayObject *)arr.get(), access_flags, true);
  }
  else {
    stringstream ss;
    py_ref obj_tp = capture_if_not_null(PyObject_Repr((PyObject *)Py_TYPE(obj)));
    ss << "could not create a dynd array from the numpy scalar object";
    ss << " of type " << pydynd::pystring_as_string(obj_tp.get());
    throw dynd::type_error(ss.str());
  }

  return result;
}

#endif // DYND_NUMPY_INTEROP

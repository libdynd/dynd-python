//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "array_functions.hpp"
#include "arrfunc_functions.hpp"
#include "array_from_py.hpp"
#include "type_functions.hpp"
#include "utility_functions.hpp"
#include "numpy_interop.hpp"
#include "types/pyobject_type.hpp"

#include <dynd/func/assignment.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/base_dim_type.hpp>
#include <dynd/memblock/external_memory_block.hpp>
#include <dynd/array_range.hpp>
#include <dynd/type_promotion.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/base_bytes_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/view.hpp>

using namespace std;
using namespace dynd;
using namespace pydynd;

PyObject *pydynd::array_index(const dynd::nd::array &n)
{
  // Implements the nb_index slot
  switch (n.get_type().get_kind()) {
  case uint_kind:
  case sint_kind:
    return array_as_py(n, false);
  default:
    PyErr_SetString(PyExc_TypeError, "dynd array must have kind 'int'"
                                     " or 'uint' to be used as an index");
    throw exception();
  }
}

PyObject *pydynd::array_nonzero(const dynd::nd::array &n)
{
  // Implements the nonzero/conversion to boolean slot
  switch (n.get_type().value_type().get_kind()) {
  case bool_kind:
  case uint_kind:
  case sint_kind:
  case real_kind:
  case complex_kind:
    // Follow Python in not raising errors here
    if (n.as<bool>(assign_error_nocheck)) {
      Py_INCREF(Py_True);
      return Py_True;
    }
    else {
      Py_INCREF(Py_False);
      return Py_False;
    }
  case string_kind: {
    // Follow Python, return True if the string is nonempty, False otherwise
    nd::array n_eval = n.eval();
    const ndt::base_string_type *bsd =
        n_eval.get_type().extended<ndt::base_string_type>();
    const char *begin = NULL, *end = NULL;
    bsd->get_string_range(&begin, &end, n_eval.get()->metadata(),
                          n_eval.cdata());
    if (begin != end) {
      Py_INCREF(Py_True);
      return Py_True;
    }
    else {
      Py_INCREF(Py_False);
      return Py_False;
    }
  }
  case bytes_kind: {
    // Return True if there is a non-zero byte, False otherwise
    nd::array n_eval = n.eval();
    const ndt::base_bytes_type *bbd =
        n_eval.get_type().extended<ndt::base_bytes_type>();
    const char *begin = NULL, *end = NULL;
    bbd->get_bytes_range(&begin, &end, n_eval.get()->metadata(),
                         n_eval.cdata());
    while (begin != end) {
      if (*begin != 0) {
        Py_INCREF(Py_True);
        return Py_True;
      }
      else {
        ++begin;
      }
    }
    Py_INCREF(Py_False);
    return Py_False;
  }
  case datetime_kind: {
    // Dates and datetimes are never zero
    // TODO: What to do with NA value?
    Py_INCREF(Py_True);
    return Py_True;
  }
  default:
    // TODO: Implement nd.any and nd.all, mention them
    //       here like NumPy does.
    PyErr_SetString(PyExc_ValueError, "the truth value of a dynd array with "
                                      "non-scalar type is ambiguous");
    throw exception();
  }
}

PyObject *pydynd::array_int(const dynd::nd::array &n)
{
  const ndt::type &vt = n.get_type().value_type();
  switch (vt.get_kind()) {
  case bool_kind:
  case uint_kind:
  case sint_kind:
    if (vt.get_type_id() != uint64_type_id) {
      return PyLong_FromLongLong(n.as<int64_t>());
    }
    else {
      return PyLong_FromUnsignedLongLong(n.as<uint64_t>());
    }
  default:
    break;
  }
  stringstream ss;
  ss << "cannot convert dynd array of type " << n.get_type();
  ss << " to an int";
  PyErr_SetString(PyExc_ValueError, ss.str().c_str());
  throw exception();
}

PyObject *pydynd::array_float(const dynd::nd::array &n)
{
  switch (n.get_type().value_type().get_kind()) {
  case bool_kind:
  case uint_kind:
  case sint_kind:
  case real_kind:
    return PyFloat_FromDouble(n.as<double>());
  default:
    break;
  }
  stringstream ss;
  ss << "cannot convert dynd array of type " << n.get_type();
  ss << " to a float";
  PyErr_SetString(PyExc_ValueError, ss.str().c_str());
  throw exception();
}

PyObject *pydynd::array_complex(const dynd::nd::array &n)
{
  switch (n.get_type().value_type().get_kind()) {
  case bool_kind:
  case uint_kind:
  case sint_kind:
  case real_kind:
  case complex_kind: {
    dynd::complex<double> value = n.as<dynd::complex<double>>();
    return PyComplex_FromDoubles(value.real(), value.imag());
  }
  default:
    break;
  }
  stringstream ss;
  ss << "cannot convert dynd array of type " << n.get_type();
  ss << " to a complex";
  PyErr_SetString(PyExc_ValueError, ss.str().c_str());
  throw exception();
}

dynd::nd::array pydynd::array_eval(const dynd::nd::array &n)
{
  return n.eval();
}

dynd::nd::array pydynd::array_zeros(const dynd::ndt::type &d, PyObject *access)
{
  uint32_t access_flags = pyarg_creation_access_flags(access);
  nd::array n = nd::empty(d);
  n.assign(0);
  if (access_flags != 0 && (access_flags & nd::write_access_flag) == 0) {
    n.flag_as_immutable();
  }
  return n;
}

dynd::nd::array pydynd::array_zeros(PyObject *shape, const dynd::ndt::type &d,
                                    PyObject *access)
{
  uint32_t access_flags = pyarg_creation_access_flags(access);
  std::vector<intptr_t> shape_vec;
  pyobject_as_vector_intp(shape, shape_vec, true);
  nd::array n = make_strided_array(d, (int)shape_vec.size(),
                                   shape_vec.empty() ? NULL : &shape_vec[0]);
  n.assign(0);
  if (access_flags != 0 && (access_flags & nd::write_access_flag) == 0) {
    n.flag_as_immutable();
  }
  return n;
}

dynd::nd::array pydynd::array_ones(const dynd::ndt::type &d, PyObject *access)
{
  uint32_t access_flags = pyarg_creation_access_flags(access);
  nd::array n = nd::empty(d);
  n.assign(1);
  if (access_flags != 0 && (access_flags & nd::write_access_flag) == 0) {
    n.flag_as_immutable();
  }
  return n;
}

dynd::nd::array pydynd::array_ones(PyObject *shape, const dynd::ndt::type &d,
                                   PyObject *access)
{
  uint32_t access_flags = pyarg_creation_access_flags(access);
  std::vector<intptr_t> shape_vec;
  pyobject_as_vector_intp(shape, shape_vec, true);
  nd::array n = make_strided_array(d, (int)shape_vec.size(),
                                   shape_vec.empty() ? NULL : &shape_vec[0]);
  n.assign(1);
  if (access_flags != 0 && (access_flags & nd::write_access_flag) == 0) {
    n.flag_as_immutable();
  }
  return n;
}

dynd::nd::array pydynd::array_empty(const dynd::ndt::type &d, PyObject *access)
{
  uint32_t access_flags = pyarg_creation_access_flags(access);
  if (access_flags != 0 &&
      (access_flags != (nd::read_access_flag | nd::write_access_flag))) {
    throw invalid_argument("access type must be readwrite for empty array");
  }
  return nd::empty(d);
}

dynd::nd::array pydynd::array_empty(PyObject *shape, const dynd::ndt::type &d,
                                    PyObject *access)
{
  uint32_t access_flags = pyarg_creation_access_flags(access);
  if (access_flags &&
      (access_flags != (nd::read_access_flag | nd::write_access_flag))) {
    throw invalid_argument("access type must be readwrite for empty array");
  }
  std::vector<intptr_t> shape_vec;
  pyobject_as_vector_intp(shape, shape_vec, true);
  return make_strided_array(d, (int)shape_vec.size(),
                            shape_vec.empty() ? NULL : &shape_vec[0]);
}

dynd::nd::array pydynd::array_cast(const dynd::nd::array &n,
                                   const ndt::type &dt)
{
  return n.cast(dt);
}

dynd::nd::array pydynd::array_ucast(const dynd::nd::array &n,
                                    const ndt::type &dt, intptr_t replace_ndim)
{
  return n.ucast(dt, replace_ndim);
}

PyObject *pydynd::array_get_shape(const dynd::nd::array &n)
{
  if (n.is_null()) {
    PyErr_SetString(PyExc_AttributeError,
                    "Cannot access attribute of null dynd array");
    throw std::exception();
  }
  size_t ndim = n.get_type().get_ndim();
  dimvector result(ndim);
  n.get_shape(result.get());
  return intptr_array_as_tuple(ndim, result.get());
}

PyObject *pydynd::array_get_strides(const dynd::nd::array &n)
{
  if (n.is_null()) {
    PyErr_SetString(PyExc_AttributeError,
                    "Cannot access attribute of null dynd array");
    throw std::exception();
  }
  size_t ndim = n.get_type().get_ndim();
  dimvector result(ndim);
  n.get_strides(result.get());
  return intptr_array_as_tuple(ndim, result.get());
}

static void pyobject_as_irange_array(intptr_t &out_size,
                                     shortvector<irange> &out_indices,
                                     PyObject *subscript)
{
  if (!PyTuple_Check(subscript)) {
    // A single subscript
    out_size = 1;
    out_indices.init(1);
    out_indices[0] = pyobject_as_irange(subscript);
  }
  else {
    out_size = PyTuple_GET_SIZE(subscript);
    // Tuple of subscripts
    out_indices.init(out_size);
    for (Py_ssize_t i = 0; i < out_size; ++i) {
      out_indices[i] = pyobject_as_irange(PyTuple_GET_ITEM(subscript, i));
    }
  }
}

dynd::nd::array pydynd::array_getitem(const dynd::nd::array &n,
                                      PyObject *subscript)
{
  if (subscript == Py_Ellipsis) {
    return n.at_array(0, NULL);
  }
  else {
    // Convert the pyobject into an array of iranges
    intptr_t size;
    shortvector<irange> indices;
    pyobject_as_irange_array(size, indices, subscript);

    // Do an indexing operation
    return n.at_array(size, indices.get());
  }
}

void pydynd::array_setitem(const dynd::nd::array &n, PyObject *subscript,
                           PyObject *value)
{
  if (subscript == Py_Ellipsis) {
    n.assign(value);
#if PY_VERSION_HEX < 0x03000000
  }
  else if (PyInt_Check(subscript)) {
    long i = PyInt_AS_LONG(subscript);
    n(i).assign(value);
#endif // PY_VERSION_HEX < 0x03000000
  }
  else if (PyLong_Check(subscript)) {
    intptr_t i = PyLong_AsSsize_t(subscript);
    if (i == -1 && PyErr_Occurred()) {
      throw runtime_error("error converting int value");
    }
    n(i).assign(value);
  }
  else {
    intptr_t size;
    shortvector<irange> indices;
    pyobject_as_irange_array(size, indices, subscript);
    n.at_array(size, indices.get(), false).assign(value);
  }
}

nd::array pydynd::array_range(PyObject *start, PyObject *stop, PyObject *step,
                              PyObject *dt)
{
  nd::array start_nd, stop_nd, step_nd;
  ndt::type dt_nd;

  if (start != Py_None) {
    start_nd = array_from_py(start, 0, false);
  }
  else {
    start_nd = 0;
  }
  stop_nd = array_from_py(stop, 0, false);
  if (step != Py_None) {
    step_nd = array_from_py(step, 0, false);
  }
  else {
    step_nd = 1;
  }

  if (dt != Py_None) {
    dt_nd = make__type_from_pyobject(dt);
  }
  else {
    dt_nd = promote_types_arithmetic(
        start_nd.get_type(),
        promote_types_arithmetic(stop_nd.get_type(), step_nd.get_type()));
  }

  start_nd = start_nd.ucast(dt_nd).eval();
  stop_nd = stop_nd.ucast(dt_nd).eval();
  step_nd = step_nd.ucast(dt_nd).eval();

  if (!start_nd.is_scalar() || !stop_nd.is_scalar() || !step_nd.is_scalar()) {
    throw runtime_error(
        "nd::range should only be called with scalar parameters");
  }

  return nd::range(dt_nd, start_nd.cdata(), stop_nd.cdata(), step_nd.cdata());
}

dynd::nd::array pydynd::array_linspace(PyObject *start, PyObject *stop,
                                       PyObject *count, PyObject *dt)
{
  nd::array start_nd, stop_nd;
  intptr_t count_val = pyobject_as_index(count);
  start_nd = array_from_py(start, 0, false);
  stop_nd = array_from_py(stop, 0, false);
  if (dt == Py_None) {
    return nd::linspace(start_nd, stop_nd, count_val);
  }
  else {
    return nd::linspace(start_nd, stop_nd, count_val,
                        make__type_from_pyobject(dt));
  }
}

dynd::nd::array pydynd::nd_fields(const nd::array &n, PyObject *field_list)
{
  vector<std::string> selected_fields;
  pyobject_as_vector_string(field_list, selected_fields);

  // TODO: Move this implementation into dynd
  ndt::type fdt = n.get_dtype();
  if (fdt.get_kind() != struct_kind) {
    stringstream ss;
    ss << "nd.fields must be given a dynd array of 'struct' kind, not ";
    ss << fdt;
    throw runtime_error(ss.str());
  }
  const ndt::struct_type *bsd = fdt.extended<ndt::struct_type>();

  if (selected_fields.empty()) {
    throw runtime_error(
        "nd.fields requires at least one field name to be specified");
  }
  // Construct the field mapping and output field types
  vector<intptr_t> selected_index(selected_fields.size());
  vector<ndt::type> selected__types(selected_fields.size());
  for (size_t i = 0; i != selected_fields.size(); ++i) {
    selected_index[i] = bsd->get_field_index(selected_fields[i]);
    if (selected_index[i] < 0) {
      stringstream ss;
      ss << "field name ";
      print_escaped_utf8_string(ss, selected_fields[i]);
      ss << " does not exist in dynd type " << fdt;
      throw runtime_error(ss.str());
    }
    selected__types[i] = bsd->get_field_type(selected_index[i]);
  }
  // Create the result udt
  ndt::type rudt = ndt::struct_type::make(selected_fields, selected__types);
  ndt::type result_tp = n.get_type().with_replaced_dtype(rudt);
  const ndt::struct_type *rudt_bsd = rudt.extended<ndt::struct_type>();

  // Allocate the new memory block.
  size_t arrmeta_size = result_tp.get_arrmeta_size();
  nd::array result(reinterpret_cast<array_preamble *>(
                       make_array_memory_block(arrmeta_size).get()),
                   true);

  // Clone the data pointer
  result.get()->data = n.get()->data;
  result.get()->owner = n.get()->owner;
  if (!result.get()->owner) {
    result.get()->owner = n.get();
  }

  // Copy the flags
  result.get()->flags = n.get()->flags;

  // Set the type and transform the arrmeta
  result.get()->tp = result_tp;
  // First copy all the array data type arrmeta
  ndt::type tmp_dt = result_tp;
  char *dst_arrmeta = result.get()->metadata();
  const char *src_arrmeta = n.get()->metadata();
  while (tmp_dt.get_ndim() > 0) {
    if (tmp_dt.get_kind() != dim_kind) {
      throw runtime_error(
          "nd.fields doesn't support dimensions with pointers yet");
    }
    const ndt::base_dim_type *budd = tmp_dt.extended<ndt::base_dim_type>();
    size_t offset = budd->arrmeta_copy_construct_onedim(
        dst_arrmeta, src_arrmeta,
        intrusive_ptr<memory_block_data>(n.get(), true));
    dst_arrmeta += offset;
    src_arrmeta += offset;
    tmp_dt = budd->get_element_type();
  }
  // Then create the arrmeta for the new struct
  const size_t *arrmeta_offsets = bsd->get_arrmeta_offsets_raw();
  const size_t *result_arrmeta_offsets = rudt_bsd->get_arrmeta_offsets_raw();
  const size_t *data_offsets = bsd->get_data_offsets(src_arrmeta);
  size_t *result_data_offsets = reinterpret_cast<size_t *>(dst_arrmeta);
  for (size_t i = 0; i != selected_fields.size(); ++i) {
    const ndt::type &dt = selected__types[i];
    // Copy the data offset
    result_data_offsets[i] = data_offsets[selected_index[i]];
    // Copy the arrmeta for this field
    if (dt.get_arrmeta_size() > 0) {
      dt.extended()->arrmeta_copy_construct(
          dst_arrmeta + result_arrmeta_offsets[i],
          src_arrmeta + arrmeta_offsets[selected_index[i]],
          intrusive_ptr<memory_block_data>(n.get(), true));
    }
  }

  return result;
}

PYDYND_API dynd::nd::array pydynd::pyobject_array(PyObject *obj)
{
  dynd::nd::array a = dynd::nd::empty(dynd::ndt::make_type<pyobject_type>());
  *reinterpret_cast<PyObject **>(a.data()) = obj;

  return a;
}

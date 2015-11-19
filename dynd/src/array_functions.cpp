//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "array_functions.hpp"
#include "arrfunc_functions.hpp"
#include "array_from_py.hpp"
#include "array_assign_from_py.hpp"
#include "type_functions.hpp"
#include "utility_functions.hpp"
#include "numpy_interop.hpp"

#include <dynd/types/string_type.hpp>
#include <dynd/types/base_dim_type.hpp>
#include <dynd/memblock/external_memory_block.hpp>
#include <dynd/array_range.hpp>
#include <dynd/type_promotion.hpp>
#include <dynd/types/base_struct_type.hpp>
#include <dynd/types/base_bytes_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/view.hpp>

using namespace std;
using namespace dynd;
using namespace pydynd;

PyObject *pydynd::array_str(const dynd::nd::array &n)
{
#if PY_VERSION_HEX >= 0x03000000
  // In Python 3, str is unicode
  if (n.is_null()) {
    return PyUnicode_FromString("nd.array()");
  }
  return array_unicode(n);
#else
  if (n.is_null()) {
    return PyString_FromString("nd.array()");
  }
  nd::array n_str;
  if (n.get_type().get_kind() == string_kind &&
      n.get_type().extended<ndt::base_string_type>()->get_encoding() ==
          string_encoding_ascii) {
    // If it's already an ASCII string, pass-through
    n_str = n;
  } else {
    // Otherwise, convert to an ASCII string
    n_str = nd::empty(ndt::string_type::make());
    n_str.vals() = n;
  }
  const ndt::base_string_type *bsd =
      n_str.get_type().extended<ndt::base_string_type>();
  const char *begin = NULL, *end = NULL;
  bsd->get_string_range(&begin, &end, n_str.get()->metadata(), n_str.cdata());
  return PyString_FromStringAndSize(begin, end - begin);
#endif
}

#if PY_VERSION_HEX >= 0x03030000
#define DYND_PY_ENCODING (string_encoding_utf_8)
#else
#if Py_UNICODE_SIZE == 2
#define DYND_PY_ENCODING (string_encoding_ucs_2)
#else
#define DYND_PY_ENCODING (string_encoding_utf_32)
#endif
#endif

PyObject *pydynd::array_unicode(const dynd::nd::array &n)
{
  nd::array n_str;
  if (n.get_type().get_kind() == string_kind &&
      n.get_type().extended<ndt::base_string_type>()->get_encoding() ==
          DYND_PY_ENCODING) {
    // If it's already a unicode string, pass-through
    n_str = n;
  } else {
    // Otherwise, convert to a unicode string
    n_str = nd::empty(ndt::string_type::make());
    n_str.vals() = n;
  }
  const ndt::base_string_type *bsd =
      static_cast<const ndt::base_string_type *>(n_str.get_type().extended());
  const char *begin = NULL, *end = NULL;
  bsd->get_string_range(&begin, &end, n_str.get()->metadata(), n_str.cdata());
#if PY_VERSION_HEX >= 0x03030000
  // TODO: Might be more efficient to use a different Python 3 API,
  //       avoiding the creation of intermediate UTF-8
  return PyUnicode_FromStringAndSize(begin, end - begin);
#else
  return PyUnicode_FromUnicode(reinterpret_cast<const Py_UNICODE *>(begin),
                               (end - begin) / sizeof(Py_UNICODE));
#endif
}

#undef DYND_PY_ENCODING

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
    } else {
      Py_INCREF(Py_False);
      return Py_False;
    }
  case string_kind: {
    // Follow Python, return True if the string is nonempty, False otherwise
    nd::array n_eval = n.eval();
    const ndt::base_string_type *bsd =
        n_eval.get_type().extended<ndt::base_string_type>();
    const char *begin = NULL, *end = NULL;
    bsd->get_string_range(&begin, &end, n_eval.get()->metadata(), n_eval.cdata());
    if (begin != end) {
      Py_INCREF(Py_True);
      return Py_True;
    } else {
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
    bbd->get_bytes_range(&begin, &end, n_eval.get()->metadata(), n_eval.cdata());
    while (begin != end) {
      if (*begin != 0) {
        Py_INCREF(Py_True);
        return Py_True;
      } else {
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
    } else {
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

void pydynd::array_init_from_pyobject(dynd::nd::array &n, PyObject *obj,
                                      PyObject *dt, bool fulltype,
                                      PyObject *access)
{
  uint32_t access_flags = 0;
  if (access != Py_None) {
    access_flags = pyarg_strings_to_int(
        access, "access", 0, "readwrite",
        nd::read_access_flag | nd::write_access_flag, "rw",
        nd::read_access_flag | nd::write_access_flag, "readonly",
        nd::read_access_flag, "r", nd::read_access_flag, "immutable",
        nd::read_access_flag | nd::immutable_access_flag);
  }
  n = array_from_py(obj, make__type_from_pyobject(dt), fulltype, access_flags,
                    &eval::default_eval_context);
}

void pydynd::array_init_from_pyobject(dynd::nd::array &n, PyObject *obj,
                                      PyObject *access)
{
  uint32_t access_flags = 0;
  if (access != Py_None) {
    access_flags = pyarg_strings_to_int(
        access, "access", 0, "readwrite",
        nd::read_access_flag | nd::write_access_flag, "rw",
        nd::read_access_flag | nd::write_access_flag, "readonly",
        nd::read_access_flag, "r", nd::read_access_flag, "immutable",
        nd::read_access_flag | nd::immutable_access_flag);
  }
  n = array_from_py(obj, access_flags, true, &eval::default_eval_context);
}

dynd::nd::array pydynd::array_view(PyObject *obj, PyObject *type,
                                   PyObject *access)
{
  uint32_t access_flags = 0;
  if (access != Py_None) {
    access_flags = pyarg_strings_to_int(
        access, "access", 0, "readwrite",
        nd::read_access_flag | nd::write_access_flag, "rw",
        nd::read_access_flag | nd::write_access_flag, "readonly",
        nd::read_access_flag, "r", nd::read_access_flag, "immutable",
        nd::read_access_flag | nd::immutable_access_flag);
  }

  // If it's a Cython w_array
  if (DyND_PyArray_Check(obj)) {
    const nd::array &obj_dynd = ((DyND_PyArrayObject *)obj)->v;
    if (access_flags != 0) {
      uint32_t raf = obj_dynd.get_access_flags();
      if ((access_flags & nd::immutable_access_flag) &&
          !(raf & nd::immutable_access_flag)) {
        throw runtime_error(
            "cannot view a non-immutable dynd array as immutable");
      }
      if ((access_flags & nd::write_access_flag) != 0 &&
          (raf & nd::write_access_flag) == 0) {
        throw runtime_error("cannot view a readonly dynd array as readwrite");
      }
      if ((access_flags & nd::write_access_flag) == 0 &&
          (raf & nd::write_access_flag) != 0) {
        // Convert it to a readonly view
        nd::array result(shallow_copy_array_memory_block(obj_dynd));
        result.get()->flags = access_flags;
        return result;
      }
    }
    if (type == Py_None) {
      return obj_dynd;
    } else {
      ndt::type tp = make__type_from_pyobject(type);
      return nd::view(obj_dynd, tp);
    }
  }

  // If it's a numpy array
  if (PyArray_Check(obj)) {
    nd::array result =
        array_from_numpy_array((PyArrayObject *)obj, access_flags, false);
    if (type == Py_None) {
      return result;
    } else {
      ndt::type tp = make__type_from_pyobject(type);
      return nd::view(result, tp);
    }
  }

  // TODO: add python buffer protocol support here
  stringstream ss;
  pyobject_ownref obj_tp(PyObject_Repr((PyObject *)Py_TYPE(obj)));
  ss << "object of type " << pystring_as_string(obj_tp.get());
  ss << " can't be viewed as a dynd array, use nd.asarray or";
  ss << " nd.array to create a copy";
  throw runtime_error(ss.str());
}

dynd::nd::array pydynd::array_asarray(PyObject *obj, PyObject *access)
{
  uint32_t access_flags = 0;
  if (access != Py_None) {
    access_flags = pyarg_strings_to_int(
        access, "access", 0, "readwrite",
        nd::read_access_flag | nd::write_access_flag, "rw",
        nd::read_access_flag | nd::write_access_flag, "readonly",
        nd::read_access_flag, "r", nd::read_access_flag, "immutable",
        nd::read_access_flag | nd::immutable_access_flag);
  }

  // If it's a dynd-native w_array
  if (DyND_PyArray_Check(obj)) {
    const nd::array &obj_dynd = ((DyND_PyArrayObject *)obj)->v;
    if (access_flags != 0) {
      // Flag for whether it's ok to take this view
      bool ok = true;
      // TODO: Make an nd::view function to handle this logic
      uint32_t raf = obj_dynd.get_access_flags();
      if ((access_flags & nd::immutable_access_flag) &&
          !(raf & nd::immutable_access_flag)) {
        ok = false;
      } else if ((access_flags & nd::write_access_flag) == 0 &&
                 (raf & nd::write_access_flag) != 0) {
        // Convert it to a readonly view
        nd::array result(shallow_copy_array_memory_block(obj_dynd));
        result.get()->flags = access_flags;
        return result;
      }
      if ((access_flags & nd::write_access_flag) != 0 &&
          (raf & nd::write_access_flag) == 0) {
        ok = false;
      }

      if (ok) {
        return obj_dynd;
      } else {
        return obj_dynd.eval_copy(access_flags);
      }
    } else {
      return obj_dynd;
    }
  }

  // If it's a numpy array
  if (PyArray_Check(obj)) {
    nd::array result =
        array_from_numpy_array((PyArrayObject *)obj, access_flags, false);
    if (access_flags == 0) {
      // Always return it as a view if no specific access flags are specified
      return result;
    } else {
      bool ok = true;
      // TODO: Make an nd::view function to handle this logic
      uint32_t raf = result.get_access_flags();
      if ((access_flags & nd::write_access_flag) != 0 &&
          (raf & nd::write_access_flag) == 0) {
        ok = false;
      }
      if ((access_flags & nd::write_access_flag) == 0 &&
          (raf & nd::write_access_flag) != 0) {
        // Convert it to a readonly view
        nd::array ro_view(shallow_copy_array_memory_block(result));
        ro_view.get()->flags = access_flags;
        return ro_view;
      }
      if (ok) {
        return result;
      } else {
        return result.eval_copy(access_flags);
      }
    }
  }

  // TODO: Check for the python buffer protocol.

  return array_from_py(obj, access_flags, true, &eval::default_eval_context);
}

dynd::nd::array pydynd::array_eval(const dynd::nd::array &n, PyObject *ectx_obj)
{
  return n.eval(eval_context_from_pyobj(ectx_obj));
}

dynd::nd::array pydynd::array_eval_copy(const dynd::nd::array &n,
                                        PyObject *access, PyObject *ectx_obj)
{
  uint32_t access_flags = pyarg_creation_access_flags(access);
  return n.eval_copy(access_flags, eval_context_from_pyobj(ectx_obj));
}

dynd::nd::array pydynd::array_zeros(const dynd::ndt::type &d, PyObject *access)
{
  uint32_t access_flags = pyarg_creation_access_flags(access);
  nd::array n = nd::empty(d);
  n.val_assign(0);
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
  nd::array n = nd::make_strided_array(
      d, (int)shape_vec.size(), shape_vec.empty() ? NULL : &shape_vec[0]);
  n.val_assign(0);
  if (access_flags != 0 && (access_flags & nd::write_access_flag) == 0) {
    n.flag_as_immutable();
  }
  return n;
}

dynd::nd::array pydynd::array_ones(const dynd::ndt::type &d, PyObject *access)
{
  uint32_t access_flags = pyarg_creation_access_flags(access);
  nd::array n = nd::empty(d);
  n.val_assign(1);
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
  nd::array n = nd::make_strided_array(
      d, (int)shape_vec.size(), shape_vec.empty() ? NULL : &shape_vec[0]);
  n.val_assign(1);
  if (access_flags != 0 && (access_flags & nd::write_access_flag) == 0) {
    n.flag_as_immutable();
  }
  return n;
}

dynd::nd::array pydynd::array_full(const dynd::ndt::type &d, PyObject *value,
                                   PyObject *access)
{
  uint32_t access_flags = pyarg_creation_access_flags(access);
  nd::array n = nd::empty(d);
  array_broadcast_assign_from_py(n, value, &eval::default_eval_context);
  if (access_flags != 0 && (access_flags & nd::write_access_flag) == 0) {
    n.flag_as_immutable();
  }
  return n;
}

dynd::nd::array pydynd::array_full(PyObject *shape, const dynd::ndt::type &d,
                                   PyObject *value, PyObject *access)
{
  uint32_t access_flags = pyarg_creation_access_flags(access);
  std::vector<intptr_t> shape_vec;
  pyobject_as_vector_intp(shape, shape_vec, true);
  nd::array n = nd::make_strided_array(
      d, (int)shape_vec.size(), shape_vec.empty() ? NULL : &shape_vec[0]);
  array_broadcast_assign_from_py(n, value, &eval::default_eval_context);
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
  return nd::make_strided_array(d, (int)shape_vec.size(),
                                shape_vec.empty() ? NULL : &shape_vec[0]);
}

dynd::nd::array pydynd::array_memmap(PyObject *filename, PyObject *begin,
                                     PyObject *end, PyObject *access)
{
  std::string filename_ = pystring_as_string(filename);
  intptr_t begin_ = (begin == Py_None) ? 0 : pyobject_as_index(begin);
  intptr_t end_ = (end == Py_None) ? std::numeric_limits<intptr_t>::max()
                                   : pyobject_as_index(end);
  uint32_t access_flags = 0;
  if (access != Py_None) {
    access_flags = pyarg_strings_to_int(
        access, "access", 0, "readwrite",
        nd::read_access_flag | nd::write_access_flag, "rw",
        nd::read_access_flag | nd::write_access_flag, "readonly",
        nd::read_access_flag, "r", nd::read_access_flag, "immutable",
        nd::read_access_flag | nd::immutable_access_flag);
  }
  return nd::memmap(filename_, begin_, end_, access_flags);
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

PyObject *pydynd::array_adapt(PyObject *a, PyObject *tp_obj, PyObject *adapt_op)
{
  return DyND_PyWrapper_New(
      array_from_py(a, 0, false, &eval::default_eval_context).adapt(
          make__type_from_pyobject(tp_obj), pystring_as_string(adapt_op)));
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

bool pydynd::array_is_scalar(const dynd::nd::array &n)
{
  if (n.is_null()) {
    PyErr_SetString(PyExc_AttributeError,
                    "Cannot access attribute of null dynd array");
    throw std::exception();
  }

  return n.is_scalar();
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
  } else {
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
  } else {
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
    array_broadcast_assign_from_py(n, value, &eval::default_eval_context);
#if PY_VERSION_HEX < 0x03000000
  } else if (PyInt_Check(subscript)) {
    long i = PyInt_AS_LONG(subscript);
    const char *arrmeta = n.get()->metadata();
    char *data = n.data();
    ndt::type d =
        n.get_type().at_single(i, &arrmeta, const_cast<const char **>(&data));
    array_broadcast_assign_from_py(d, arrmeta, data, value,
                                   &eval::default_eval_context);
#endif // PY_VERSION_HEX < 0x03000000
  } else if (PyLong_Check(subscript)) {
    intptr_t i = PyLong_AsSsize_t(subscript);
    if (i == -1 && PyErr_Occurred()) {
      throw runtime_error("error converting int value");
    }
    const char *arrmeta = n.get()->metadata();
    char *data = n.data();
    ndt::type d =
        n.get_type().at_single(i, &arrmeta, const_cast<const char **>(&data));
    array_broadcast_assign_from_py(d, arrmeta, data, value,
                                   &eval::default_eval_context);
  } else {
    intptr_t size;
    shortvector<irange> indices;
    pyobject_as_irange_array(size, indices, subscript);
    array_broadcast_assign_from_py(n.at_array(size, indices.get(), false),
                                   value, &eval::default_eval_context);
  }
}

nd::array pydynd::array_range(PyObject *start, PyObject *stop, PyObject *step,
                              PyObject *dt)
{
  nd::array start_nd, stop_nd, step_nd;
  ndt::type dt_nd;

  if (start != Py_None) {
    start_nd = array_from_py(start, 0, false, &eval::default_eval_context);
  } else {
    start_nd = 0;
  }
  stop_nd = array_from_py(stop, 0, false, &eval::default_eval_context);
  if (step != Py_None) {
    step_nd = array_from_py(step, 0, false, &eval::default_eval_context);
  } else {
    step_nd = 1;
  }

  if (dt != Py_None) {
    dt_nd = make__type_from_pyobject(dt);
  } else {
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
  start_nd = array_from_py(start, 0, false, &eval::default_eval_context);
  stop_nd = array_from_py(stop, 0, false, &eval::default_eval_context);
  if (dt == Py_None) {
    return nd::linspace(start_nd, stop_nd, count_val);
  } else {
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
  const ndt::base_struct_type *bsd = fdt.extended<ndt::base_struct_type>();

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
  const ndt::base_struct_type *rudt_bsd =
      rudt.extended<ndt::base_struct_type>();

  // Allocate the new memory block.
  size_t arrmeta_size = result_tp.get_arrmeta_size();
  nd::array result(make_array_memory_block(arrmeta_size));

  // Clone the data pointer
  result.get()->data = n.get()->data;
  result.get()->owner = n.get()->owner;
  if (!result.get()->owner) {
    result.get()->owner = n;
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
    size_t offset =
        budd->arrmeta_copy_construct_onedim(dst_arrmeta, src_arrmeta, n);
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
          src_arrmeta + arrmeta_offsets[selected_index[i]], n);
    }
  }

  return result;
}

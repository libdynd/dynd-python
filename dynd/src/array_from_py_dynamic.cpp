//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>
#include <datetime.h>

#include <dynd/types/string_type.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/base_struct_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/callable_type.hpp>
#include <dynd/types/datetime_type.hpp>
#include <dynd/types/type_type.hpp>
#include <dynd/type_promotion.hpp>
#include <dynd/memblock/external_memory_block.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/type_promotion.hpp>
#include <dynd/exceptions.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

#include "array_from_py.hpp"
#include "array_from_py_typededuction.hpp"
#include "array_from_py_dynamic.hpp"
#include "array_assign_from_py.hpp"
#include "array_functions.hpp"
#include "type_functions.hpp"
#include "utility_functions.hpp"
#include "numpy_interop.hpp"

using namespace std;
using namespace dynd;
using namespace pydynd;

static const intptr_t ARRAY_FROM_DYNAMIC_INITIAL_COUNT = 16;

void pydynd::init_array_from_py_dynamic()
{
  // Initialize the pydatetime API
  PyDateTime_IMPORT;
}

namespace {

struct afpd_coordentry {
  // The current coordinate of this axis being processed
  intptr_t coord;
  // The type in the output array for this axis
  ndt::type tp;
  // The arrmeta pointer in the output array for this axis
  const char *arrmeta_ptr;
  // The data pointer in the output array for the next axis (or element)
  char *data_ptr;
  // Used for var dimensions, the amount of presently reserved space
  intptr_t reserved_size;
};

struct afpd_dtype {
  // The data type after all the dimensions
  ndt::type dtp;
  // The arrmeta pointer in the output array for the dtype
  const char *arrmeta_ptr;

  void swap(afpd_dtype &rhs)
  {
    dtp.swap(rhs.dtp);
    std::swap(arrmeta_ptr, rhs.arrmeta_ptr);
  }
};

} // anonymous namespace

static void array_from_py_dynamic(PyObject *obj, std::vector<intptr_t> &shape,
                                  std::vector<afpd_coordentry> &coord,
                                  afpd_dtype &elem, dynd::nd::array &arr,
                                  intptr_t current_axis,
                                  const eval::eval_context *ectx);

/**
 * This allocates an nd::array for the first time,
 * using the shape provided, and filling in the
 * `coord` and `elem`.
 *
 * Pass shape.size() to promoted_axis for initial
 * allocations or dtype promotions.
 */
static nd::array allocate_nd_arr(const std::vector<intptr_t> &shape,
                                 std::vector<afpd_coordentry> &coord,
                                 afpd_dtype &elem, intptr_t promoted_axis)
{
  intptr_t ndim = (intptr_t)shape.size();
  // Allocate the nd::array
  nd::array result =
      nd::make_strided_array(elem.dtp, ndim, ndim == 0 ? NULL : &shape[0]);
  // Fill `coord` with pointers from the allocated arrays,
  // reserving some data for any var dimensions.
  coord.resize(ndim);
  ndt::type tp = result.get_type();
  const char *arrmeta_ptr = result.get()->metadata();
  char *data_ptr = result.data();
  for (intptr_t i = 0; i < ndim; ++i) {
    afpd_coordentry &c = coord[i];
    c.coord = 0;
    c.tp = tp;
    c.arrmeta_ptr = arrmeta_ptr;
    // If it's a var dim, reserve some space
    if (tp.get_type_id() == var_dim_type_id) {
      if (i < promoted_axis) {
        // Only initialize the var dim elements prior
        // to the promoted axis
        intptr_t initial_count = ARRAY_FROM_DYNAMIC_INITIAL_COUNT;
        ndt::var_dim_element_initialize(tp, arrmeta_ptr, data_ptr,
                                        initial_count);
        c.reserved_size = initial_count;
        data_ptr = reinterpret_cast<const var_dim_type_data *>(data_ptr)->begin;
      } else {
        data_ptr = NULL;
      }
      // Advance arrmeta_ptr and data_ptr to the child dimension
      arrmeta_ptr += sizeof(var_dim_type_arrmeta);
      tp = tp.extended<ndt::var_dim_type>()->get_element_type();
    } else {
      // Advance arrmeta_ptr and data_ptr to the child dimension
      arrmeta_ptr += sizeof(fixed_dim_type_arrmeta);
      tp = tp.extended<ndt::fixed_dim_type>()->get_element_type();
    }
    c.data_ptr = data_ptr;
  }
  elem.arrmeta_ptr = arrmeta_ptr;

  return result;
}

/**
 * This copies everything up to, and possibly including, the
 * current coordinate in `src_coord`. When finished,
 * `dst_coord` is left in a state equivalent to `src_coord`.
 *
 * The `copy_final_coord` controls whether the final coordinate
 * for the axis `current_axis - 1` gets copied or not. Generally
 * it shouldn't be, but when dealing with iterators it becomes
 * necessary so as to preserve the already retrieved data.
 *
 * If `promoted_axis` is less than shape.size(), it's for
 * the case where a dim was promoted from strided to var.
 * If `promoted_axis` is equal to shape.size(), it's for
 * promotion of a dtype.
 */
static void copy_to_promoted_nd_arr(
    const std::vector<intptr_t> &shape, char *dst_data_ptr,
    std::vector<afpd_coordentry> &dst_coord, afpd_dtype &dst_elem,
    const char *src_data_ptr, std::vector<afpd_coordentry> &src_coord,
    afpd_dtype &src_elem, const ckernel_builder<kernel_request_host> &ck,
    intptr_t current_axis, intptr_t promoted_axis, bool copy_final_coord,
    bool final_coordinate)
{
  if (current_axis == promoted_axis - 1) {
    // Base case - the final dimension
    if (shape[current_axis] >= 0) {
      // fixed dimension case
      const fixed_dim_type_arrmeta *dst_md =
          reinterpret_cast<const fixed_dim_type_arrmeta *>(
              dst_coord[current_axis].arrmeta_ptr);
      const fixed_dim_type_arrmeta *src_md =
          reinterpret_cast<const fixed_dim_type_arrmeta *>(
              src_coord[current_axis].arrmeta_ptr);
      if (!final_coordinate) {
        expr_strided_t fn = ck.get()->get_function<expr_strided_t>();
        // Copy the full dimension
        char *src = const_cast<char *>(src_data_ptr);
        fn(ck.get(), dst_data_ptr, dst_md->stride, &src, &src_md->stride,
           shape[current_axis]);
      } else {
        expr_strided_t fn = ck.get()->get_function<expr_strided_t>();
        // Copy up to, and possibly including, the coordinate
        char *src = const_cast<char *>(src_data_ptr);
        fn(ck.get(), dst_data_ptr, dst_md->stride, &src, &src_md->stride,
           src_coord[current_axis].coord + int(copy_final_coord));
        dst_coord[current_axis].coord = src_coord[current_axis].coord;
        dst_coord[current_axis].data_ptr =
            dst_data_ptr + dst_md->stride * dst_coord[current_axis].coord;
      }
    } else {
      // var dimension case
      const var_dim_type_arrmeta *dst_md =
          reinterpret_cast<const var_dim_type_arrmeta *>(
              dst_coord[current_axis].arrmeta_ptr);
      const var_dim_type_arrmeta *src_md =
          reinterpret_cast<const var_dim_type_arrmeta *>(
              src_coord[current_axis].arrmeta_ptr);
      var_dim_type_data *dst_d =
          reinterpret_cast<var_dim_type_data *>(dst_data_ptr);
      const var_dim_type_data *src_d =
          reinterpret_cast<const var_dim_type_data *>(src_data_ptr);
      if (!final_coordinate) {
        ndt::var_dim_element_resize(dst_coord[current_axis].tp,
                                    dst_coord[current_axis].arrmeta_ptr,
                                    dst_data_ptr, src_d->size);
        expr_strided_t fn = ck.get()->get_function<expr_strided_t>();
        // Copy the full dimension
        char *src = src_d->begin;
        fn(ck.get(), dst_d->begin, dst_md->stride, &src, &src_md->stride,
           src_d->size);
      } else {
        // Initialize the var element to the same reserved space as the
        // input
        ndt::var_dim_element_resize(
            dst_coord[current_axis].tp, dst_coord[current_axis].arrmeta_ptr,
            dst_data_ptr, src_coord[current_axis].reserved_size);
        dst_coord[current_axis].reserved_size =
            src_coord[current_axis].reserved_size;
        // Copy up to, and possibly including, the coordinate
        expr_strided_t fn = ck.get()->get_function<expr_strided_t>();
        if (fn != NULL) {
          char *src = src_d->begin;
          fn(ck.get(), dst_d->begin, dst_md->stride, &src, &src_md->stride,
             src_coord[current_axis].coord + int(copy_final_coord));
        }
        dst_coord[current_axis].coord = src_coord[current_axis].coord;
        dst_coord[current_axis].data_ptr =
            dst_d->begin + dst_md->stride * dst_coord[current_axis].coord;
      }
    }
  } else {
    // Recursive case
    if (shape[current_axis] >= 0) {
      // strided dimension case
      const fixed_dim_type_arrmeta *dst_md =
          reinterpret_cast<const fixed_dim_type_arrmeta *>(
              dst_coord[current_axis].arrmeta_ptr);
      const fixed_dim_type_arrmeta *src_md =
          reinterpret_cast<const fixed_dim_type_arrmeta *>(
              src_coord[current_axis].arrmeta_ptr);
      if (!final_coordinate) {
        // Copy the full dimension
        intptr_t size = shape[current_axis];
        intptr_t dst_stride = dst_md->stride;
        intptr_t src_stride = src_md->stride;
        for (intptr_t i = 0; i < size;
             ++i, dst_data_ptr += dst_stride, src_data_ptr += src_stride) {
          copy_to_promoted_nd_arr(shape, dst_data_ptr, dst_coord, dst_elem,
                                  src_data_ptr, src_coord, src_elem, ck,
                                  current_axis + 1, promoted_axis,
                                  copy_final_coord, false);
        }
      } else {
        // Copy up to, and including, the coordinate
        intptr_t size = src_coord[current_axis].coord;
        intptr_t dst_stride = dst_md->stride;
        intptr_t src_stride = src_md->stride;
        dst_coord[current_axis].coord = size;
        dst_coord[current_axis].data_ptr = dst_data_ptr + dst_stride * size;
        for (intptr_t i = 0; i <= size;
             ++i, dst_data_ptr += dst_stride, src_data_ptr += src_stride) {
          copy_to_promoted_nd_arr(shape, dst_data_ptr, dst_coord, dst_elem,
                                  src_data_ptr, src_coord, src_elem, ck,
                                  current_axis + 1, promoted_axis,
                                  copy_final_coord, i == size);
        }
      }
    } else {
      // var dimension case
      const var_dim_type_arrmeta *dst_md =
          reinterpret_cast<const var_dim_type_arrmeta *>(
              dst_coord[current_axis].arrmeta_ptr);
      const var_dim_type_arrmeta *src_md =
          reinterpret_cast<const var_dim_type_arrmeta *>(
              src_coord[current_axis].arrmeta_ptr);
      var_dim_type_data *dst_d =
          reinterpret_cast<var_dim_type_data *>(dst_data_ptr);
      const var_dim_type_data *src_d =
          reinterpret_cast<const var_dim_type_data *>(src_data_ptr);
      if (!final_coordinate) {
        ndt::var_dim_element_resize(dst_coord[current_axis].tp,
                                    dst_coord[current_axis].arrmeta_ptr,
                                    dst_data_ptr, src_d->size);
        // Copy the full dimension
        intptr_t size = src_d->size;
        char *dst_elem_ptr = dst_d->begin;
        intptr_t dst_stride = dst_md->stride;
        const char *src_elem_ptr = src_d->begin;
        intptr_t src_stride = src_md->stride;
        for (intptr_t i = 0; i < size;
             ++i, dst_elem_ptr += dst_stride, src_elem_ptr += src_stride) {
          copy_to_promoted_nd_arr(shape, dst_elem_ptr, dst_coord, dst_elem,
                                  src_elem_ptr, src_coord, src_elem, ck,
                                  current_axis + 1, promoted_axis,
                                  copy_final_coord, false);
        }
      } else {
        // Initialize the var element to the same reserved space as the input
        ndt::var_dim_element_resize(
            dst_coord[current_axis].tp, dst_coord[current_axis].arrmeta_ptr,
            dst_data_ptr, src_coord[current_axis].reserved_size);
        dst_coord[current_axis].reserved_size =
            src_coord[current_axis].reserved_size;
        // Copy up to, and including, the size
        intptr_t size = src_coord[current_axis].coord;
        char *dst_elem_ptr = dst_d->begin;
        intptr_t dst_stride = dst_md->stride;
        const char *src_elem_ptr = src_d->begin;
        intptr_t src_stride = src_md->stride;
        dst_coord[current_axis].coord = size;
        dst_coord[current_axis].data_ptr = dst_elem_ptr + dst_stride * size;
        for (intptr_t i = 0; i <= size;
             ++i, dst_elem_ptr += dst_stride, src_elem_ptr += src_stride) {
          copy_to_promoted_nd_arr(shape, dst_elem_ptr, dst_coord, dst_elem,
                                  src_elem_ptr, src_coord, src_elem, ck,
                                  current_axis + 1, promoted_axis,
                                  copy_final_coord, i == size);
        }
      }
    }
  }
}

/**
 * This function promotes the dtype the array currently has with
 * `tp`, allocates a new one, then copies all the data up to the
 * current index in `coord`. This modifies coord and elem in place.
 */
static void promote_nd_arr_dtype(const std::vector<intptr_t> &shape,
                                 std::vector<afpd_coordentry> &coord,
                                 afpd_dtype &elem, nd::array &arr,
                                 const ndt::type &tp)
{
  intptr_t ndim = shape.size();
  vector<afpd_coordentry> newcoord;
  afpd_dtype newelem;
  if (elem.dtp.get_type_id() == uninitialized_type_id) {
    // If the `elem` dtype is uninitialized, it means a dummy
    // array was created to capture dimensional structure until
    // the first value is encountered
    newelem.dtp = tp;
  } else {
    newelem.dtp = promote_types_arithmetic(elem.dtp, tp);
  }
  // Create the new array
  nd::array newarr = allocate_nd_arr(shape, newcoord, newelem, ndim);
  // Copy the data up to, but not including, the current `coord`
  // from the old `arr` to the new one
  ckernel_builder<kernel_request_host> k;
  if (elem.dtp.get_type_id() != uninitialized_type_id) {
    make_assignment_kernel(&k, 0, newelem.dtp, newelem.arrmeta_ptr, elem.dtp,
                           elem.arrmeta_ptr, kernel_request_strided,
                           &eval::default_eval_context);
  } else {
    // An assignment kernel which copies one byte - will only
    // be called with count==0 when dtp is uninitialized
    make_assignment_kernel(&k, 0, ndt::type::make<char>(), NULL,
                           ndt::type::make<char>(), NULL,
                           kernel_request_strided, &eval::default_eval_context);
  }
  copy_to_promoted_nd_arr(shape, newarr.data(), newcoord, newelem,
                          arr.cdata(), coord, elem, k, 0, ndim,
                          false, true);
  arr.swap(newarr);
  coord.swap(newcoord);
  elem.swap(newelem);
}

/**
 * This function promotes the requested `axis` from
 * a strided dim to a var dim. It modifies `shape`, `coord`,
 * `elem`, and `arr` to point to a new array, and
 * copies the data over.
 */
static void promote_nd_arr_dim(std::vector<intptr_t> &shape,
                               std::vector<afpd_coordentry> &coord,
                               afpd_dtype &elem, nd::array &arr, intptr_t axis,
                               bool copy_final_coord)
{
  vector<afpd_coordentry> newcoord;
  afpd_dtype newelem;
  newelem.dtp = elem.dtp;
  // Convert the axis into a var dim
  shape[axis] = -1;
  // Create the new array
  nd::array newarr = allocate_nd_arr(shape, newcoord, newelem, axis);
  // Copy the data up to, but not including, the current `coord`
  // from the old `arr` to the new one. The recursion stops
  // at `axis`, where all subsequent dimensions are handled by the
  // created kernel.
  ckernel_builder<kernel_request_host> k;
  if (elem.dtp.get_type_id() != uninitialized_type_id) {
    make_assignment_kernel(&k, 0, newcoord[axis].tp, newcoord[axis].arrmeta_ptr,
                           coord[axis].tp, coord[axis].arrmeta_ptr,
                           kernel_request_strided, &eval::default_eval_context);
  }
  copy_to_promoted_nd_arr(shape, newarr.data(), newcoord, newelem,
                          arr.cdata(), coord, elem, k, 0, axis,
                          copy_final_coord, true);
  arr.swap(newarr);
  coord.swap(newcoord);
  elem.swap(newelem);
}

static bool bool_assign(char *data, PyObject *obj)
{
  if (obj == Py_True) {
    *data = 1;
    return true;
  } else if (obj == Py_False) {
    *data = 0;
    return true;
  } else {
    return false;
  }
}

static bool int_assign(const ndt::type &tp, char *data, PyObject *obj)
{
#if PY_VERSION_HEX < 0x03000000
  if (PyInt_Check(obj)) {
    long value = PyInt_AS_LONG(obj);
    // Check whether we're assigning to int32 or int64
    if (tp.get_type_id() == int64_type_id) {
      int64_t *result_ptr = reinterpret_cast<int64_t *>(data);
      *result_ptr = static_cast<int64_t>(value);
    } else {
#if SIZEOF_LONG > SIZEOF_INT
      if (value >= INT_MIN && value <= INT_MAX) {
        int32_t *result_ptr = reinterpret_cast<int32_t *>(data);
        *result_ptr = static_cast<int32_t>(value);
      } else {
        // Needs promotion to int64
        return false;
      }
    }
#else
      int32_t *result_ptr = reinterpret_cast<int32_t *>(data);
      *result_ptr = static_cast<int32_t>(value);
    }
#endif
    return true;
  }
#endif

  if (PyLong_Check(obj)) {
    PY_LONG_LONG value = PyLong_AsLongLong(obj);
    if (value == -1 && PyErr_Occurred()) {
      throw runtime_error("error converting int value");
    }

    if (tp.get_type_id() == int64_type_id) {
      int64_t *result_ptr = reinterpret_cast<int64_t *>(data);
      *result_ptr = static_cast<int64_t>(value);
    } else {
      if (value >= INT_MIN && value <= INT_MAX) {
        int32_t *result_ptr = reinterpret_cast<int32_t *>(data);
        *result_ptr = static_cast<int32_t>(value);
      } else {
        // This value requires promotion to int64
        return false;
      }
    }
    return true;
  }

  return false;
}

static bool real_assign(char *data, PyObject *obj)
{
  double *result_ptr = reinterpret_cast<double *>(data);
  if (PyFloat_Check(obj)) {
    *result_ptr = PyFloat_AS_DOUBLE(obj);
    return true;
#if PY_VERSION_HEX < 0x03000000
  } else if (PyInt_Check(obj)) {
    *result_ptr = PyInt_AS_LONG(obj);
    return true;
#endif
  } else if (PyLong_Check(obj)) {
    double value = PyLong_AsDouble(obj);
    if (value == -1 && PyErr_Occurred()) {
      throw runtime_error("error converting int value");
    }
    *result_ptr = value;
    return true;
  } else if (obj == Py_True) {
    *result_ptr = 1;
    return true;
  } else if (obj == Py_False) {
    *result_ptr = 0;
    return true;
  } else {
    return false;
  }
}

static bool complex_assign(char *data, PyObject *obj)
{
  dynd::complex<double> *result_ptr =
      reinterpret_cast<dynd::complex<double> *>(data);
  if (PyComplex_Check(obj)) {
    *result_ptr = dynd::complex<double>(PyComplex_RealAsDouble(obj),
                                        PyComplex_ImagAsDouble(obj));
    return true;
  } else if (PyFloat_Check(obj)) {
    *result_ptr = PyFloat_AS_DOUBLE(obj);
    return true;
#if PY_VERSION_HEX < 0x03000000
  } else if (PyInt_Check(obj)) {
    *result_ptr = PyInt_AS_LONG(obj);
    return true;
#endif
  } else if (PyLong_Check(obj)) {
    double value = PyLong_AsDouble(obj);
    if (value == -1 && PyErr_Occurred()) {
      throw runtime_error("error converting int value");
    }
    *result_ptr = value;
    return true;
  } else if (obj == Py_True) {
    *result_ptr = 1;
    return true;
  } else if (obj == Py_False) {
    *result_ptr = 0;
    return true;
  } else {
    return false;
  }
}

/**
 * Assign a string pyobject to an array element.
 *
 * \param  tp  The string type.
 * \param  arrmeta  Arrmeta for the string.
 * \param  data  The data element being assigned to.
 * \param  obj  The Python object containing the value
 *
 * \return  True if the assignment was successful, false if the input type is
 * incompatible.
 */
static bool string_assign(const ndt::type &tp, const char *arrmeta, char *data,
                          PyObject *obj, const eval::eval_context *ectx)
{
  if (PyUnicode_Check(obj)) {
    // Go through UTF8 (was accessing the cpython unicode values directly
    // before, but on Python 3.3 OS X it didn't work correctly.)
    pyobject_ownref utf8(PyUnicode_AsUTF8String(obj));
    char *s = NULL;
    Py_ssize_t len = 0;
    if (PyBytes_AsStringAndSize(utf8.get(), &s, &len) < 0) {
      throw exception();
    }

    const ndt::string_type *st = tp.extended<ndt::string_type>();
    st->set_from_utf8_string(arrmeta, data, s, s + len, ectx);
    return true;
  }
#if PY_VERSION_HEX < 0x03000000
  else if (PyString_Check(obj)) {
    char *s = NULL;
    intptr_t len = 0;
    if (PyString_AsStringAndSize(obj, &s, &len) < 0) {
      throw runtime_error("Error getting string data");
    }

    const ndt::string_type *st = tp.extended<ndt::string_type>();
    st->set_from_utf8_string(arrmeta, data, s, s + len, ectx);
    return true;
  }
#endif
  else {
    return false;
  }
}

#if PY_VERSION_HEX >= 0x03000000
/**
 * Assign a bytes pyobject to an array element.
 *
 * \param  tp  The string type.
 * \param  arrmeta  Arrmeta for the type.
 * \param  data  The data element being assigned to.
 * \param  obj  The Python object containing the value
 *
 * \return  True if the assignment was successful, false if the input type is
 *          incompatible.
 */
static bool bytes_assign(const ndt::type &tp, const char *arrmeta, char *data,
                         PyObject *obj)
{
  if (PyBytes_Check(obj)) {
    char *s = NULL;
    intptr_t len = 0;
    if (PyBytes_AsStringAndSize(obj, &s, &len) < 0) {
      throw runtime_error("Error getting bytes data");
    }

    const ndt::bytes_type *st = tp.extended<ndt::bytes_type>();
    st->set_bytes_data(arrmeta, data, s, s + len);
    return true;
  } else {
    return false;
  }
}
#endif

static void array_from_py_dynamic_first_alloc(
    PyObject *obj, std::vector<intptr_t> &shape,
    std::vector<afpd_coordentry> &coord, afpd_dtype &elem, dynd::nd::array &arr,
    intptr_t current_axis, const eval::eval_context *ectx)
{
  if (PyUnicode_Check(obj)
#if PY_VERSION_HEX < 0x03000000
      || PyString_Check(obj)
#endif
      ) {
    // Special case strings, because they act as sequences too
    elem.dtp = ndt::string_type::make();
    arr = allocate_nd_arr(shape, coord, elem, shape.size());
    string_assign(elem.dtp, elem.arrmeta_ptr, coord[current_axis - 1].data_ptr,
                  obj, ectx);
    return;
  }

#if PY_VERSION_HEX >= 0x03000000
  if (PyBytes_Check(obj)) {
    // Special case bytes, because they act as sequences too
    elem.dtp = ndt::bytes_type::make(1);
    arr = allocate_nd_arr(shape, coord, elem, shape.size());
    bytes_assign(elem.dtp, elem.arrmeta_ptr, coord[current_axis - 1].data_ptr,
                 obj);
    return;
  }
#endif

  if (PyDict_Check(obj)) {
    throw invalid_argument("cannot automatically deduce dynd type to "
                           "convert dict into nd.array");
  }

  if (PySequence_Check(obj)) {
    Py_ssize_t size = PySequence_Size(obj);
    if (size != -1) {
      // Add this size to the shape
      shape.push_back(size);
      // Initialize the data pointer for child elements with the one for
      // this element
      if (!coord.empty() && current_axis > 0) {
        coord[current_axis].data_ptr = coord[current_axis - 1].data_ptr;
      }
      // Process all the elements
      for (intptr_t i = 0; i < size; ++i) {
        if (!coord.empty()) {
          coord[current_axis].coord = i;
        }
        pyobject_ownref item(PySequence_GetItem(obj, i));
        array_from_py_dynamic(item.get(), shape, coord, elem, arr,
                              current_axis + 1, ectx);
        // Advance to the next element. Because the array may be
        // dynamically reallocated deeper in the recursive call, we
        // need to get the stride from the arrmeta each time.
        const fixed_dim_type_arrmeta *md =
            reinterpret_cast<const fixed_dim_type_arrmeta *>(
                coord[current_axis].arrmeta_ptr);
        coord[current_axis].data_ptr += md->stride;
      }

      if (size == 0) {
        // When the sequence is zero-sized, we can't
        // start deducing the type yet. To start capturing the
        // dimensional structure, we make an array of
        // int32, while keeping elem.dtp as an uninitialized type.
        elem.dtp = ndt::type::make<int32_t>();
        arr = allocate_nd_arr(shape, coord, elem, shape.size());
        // Make the dtype uninitialized again, to signal we have
        // deduced anything yet.
        elem.dtp = ndt::type();
      }
      return;
    } else {
      // If it doesn't actually check out as a sequence,
      // fall through eventually to the iterator check.
      PyErr_Clear();
    }
  }

  if (PyBool_Check(obj)) {
    elem.dtp = ndt::type::make<dynd::bool1>();
    arr = allocate_nd_arr(shape, coord, elem, shape.size());
    *coord[current_axis - 1].data_ptr = (obj == Py_True);
    return;
  }

#if PY_VERSION_HEX < 0x03000000
  if (PyInt_Check(obj)) {
    long value = PyInt_AS_LONG(obj);
#if SIZEOF_LONG > SIZEOF_INT
    // Use a 32-bit int if it fits.
    if (value >= INT_MIN && value <= INT_MAX) {
      elem.dtp = ndt::type::make<int32_t>();
      arr = allocate_nd_arr(shape, coord, elem, shape.size());
      int32_t *result_ptr =
          reinterpret_cast<int32_t *>(coord[current_axis - 1].data_ptr);
      *result_ptr = static_cast<int32_t>(value);
    } else {
      elem.dtp = ndt::type::make<int64_t>();
      arr = allocate_nd_arr(shape, coord, elem, shape.size());
      int64_t *result_ptr =
          reinterpret_cast<int64_t *>(coord[current_axis - 1].data_ptr);
      *result_ptr = static_cast<int64_t>(value);
    }
#else
    elem.dtp = ndt::type::make<int32_t>();
    arr = allocate_nd_arr(shape, coord, elem, shape.size());
    int32_t *result_ptr =
        reinterpret_cast<int32_t *>(coord[current_axis - 1].data_ptr);
    *result_ptr = static_cast<int32_t>(value);
#endif
    return;
  }
#endif

  if (PyLong_Check(obj)) {
    PY_LONG_LONG value = PyLong_AsLongLong(obj);
    if (value == -1 && PyErr_Occurred()) {
      throw runtime_error("error converting int value");
    }

    // Use a 32-bit int if it fits.
    if (value >= INT_MIN && value <= INT_MAX) {
      elem.dtp = ndt::type::make<int32_t>();
      arr = allocate_nd_arr(shape, coord, elem, shape.size());
      int32_t *result_ptr =
          reinterpret_cast<int32_t *>(coord[current_axis - 1].data_ptr);
      *result_ptr = static_cast<int32_t>(value);
    } else {
      elem.dtp = ndt::type::make<int64_t>();
      arr = allocate_nd_arr(shape, coord, elem, shape.size());
      int64_t *result_ptr =
          reinterpret_cast<int64_t *>(coord[current_axis - 1].data_ptr);
      *result_ptr = static_cast<int64_t>(value);
    }
    return;
  }

  if (PyFloat_Check(obj)) {
    elem.dtp = ndt::type::make<double>();
    arr = allocate_nd_arr(shape, coord, elem, shape.size());
    double *result_ptr =
        reinterpret_cast<double *>(coord[current_axis - 1].data_ptr);
    *result_ptr = PyFloat_AS_DOUBLE(obj);
    return;
  }

  if (PyComplex_Check(obj)) {
    elem.dtp = ndt::type::make<dynd::complex<double>>();
    arr = allocate_nd_arr(shape, coord, elem, shape.size());
    dynd::complex<double> *result_ptr =
        reinterpret_cast<dynd::complex<double> *>(
            coord[current_axis - 1].data_ptr);
    *result_ptr = dynd::complex<double>(PyComplex_RealAsDouble(obj),
                                        PyComplex_ImagAsDouble(obj));
    return;
  }

  // Check if it's an iterator
  {
    PyObject *iter = PyObject_GetIter(obj);
    if (iter != NULL) {
      pyobject_ownref iter_owner(iter);
      // Indicate a var dim.
      shape.push_back(-1);
      PyObject *item = PyIter_Next(iter);
      if (item != NULL) {
        intptr_t i = 0;
        while (item != NULL) {
          pyobject_ownref item_ownref(item);
          if (!coord.empty()) {
            coord[current_axis].coord = i;
            char *data_ptr = (current_axis > 0)
                                 ? coord[current_axis - 1].data_ptr
                                 : arr.data();
            if (coord[current_axis].coord >=
                coord[current_axis].reserved_size) {
              // Increase the reserved capacity if needed
              coord[current_axis].reserved_size *= 2;
              ndt::var_dim_element_resize(
                  coord[current_axis].tp, coord[current_axis].arrmeta_ptr,
                  data_ptr, coord[current_axis].reserved_size);
            }
            // Set the data pointer for the child element
            var_dim_type_data *d =
                reinterpret_cast<var_dim_type_data *>(data_ptr);
            const var_dim_type_arrmeta *md =
                reinterpret_cast<const var_dim_type_arrmeta *>(
                    coord[current_axis].arrmeta_ptr);
            coord[current_axis].data_ptr = d->begin + i * md->stride;
          }
          array_from_py_dynamic(item, shape, coord, elem, arr, current_axis + 1,
                                ectx);

          item = PyIter_Next(iter);
          ++i;
        }

        if (PyErr_Occurred()) {
          // Propagate any error
          throw exception();
        }
        // Shrink the var element to fit
        char *data_ptr =
            (current_axis > 0) ? coord[current_axis - 1].data_ptr : arr.data();
        ndt::var_dim_element_resize(coord[current_axis].tp,
                                    coord[current_axis].arrmeta_ptr, data_ptr,
                                    i);
        return;
      } else {
        // Because this iterator's sequence is zero-sized, we can't
        // start deducing the type yet. To start capturing the
        // dimensional structure, we make an array of
        // int32, while keeping elem.dtp as an uninitialized type.
        elem.dtp = ndt::type::make<int32_t>();
        arr = allocate_nd_arr(shape, coord, elem, shape.size());
        // Make the dtype uninitialized again, to signal we have
        // deduced anything yet.
        elem.dtp = ndt::type();
        // Set it to a zero-sized var element
        char *data_ptr =
            (current_axis > 0) ? coord[current_axis - 1].data_ptr : arr.data();
        ndt::var_dim_element_resize(coord[current_axis].tp,
                                    coord[current_axis].arrmeta_ptr, data_ptr,
                                    0);
        return;
      }
    } else {
      if (PyErr_ExceptionMatches(PyExc_TypeError)) {
        // A TypeError indicates that the object doesn't support
        // the iterator protocol
        PyErr_Clear();
      } else {
        // Propagate the error
        throw exception();
      }
    }
  }

  stringstream ss;
  ss << "Unable to convert object of type ";
  pyobject_ownref typestr(PyObject_Str((PyObject *)Py_TYPE(obj)));
  ss << pystring_as_string(typestr.get());
  ss << " to a dynd array";
  throw runtime_error(ss.str());
}

/**
 * Converts a Python object into an nd::array, dynamically
 * reallocating the result with a new type if a new value
 * requires it.
 *
 * When called initially, `coord` and `shape`should be empty, and
 * `arr` should be a NULL nd::array, and current_axis should be 0.
 *
 * A dimension will start as `strided` if the first element
 * encountered follows the sequence protocol, so its size is known.
 * It will start as `var` if the first element encountered follows
 * the iterator protocol.
 *
 * If a dimension is `strided`, and either a sequence of a different
 * size, or an iterator is encountered, that dimension will be
 * promoted to `var.
 *
 * \param obj  The PyObject to convert to an nd::array.
 * \param shape  A vector, same size as coord, of the array shape. Negative
 *               entries for var dimensions
 * \param coord  A vector of afpd_coordentry. This tracks the current
 *               coordinate being allocated and for var
 *               dimensions, the amount of reserved space.
 * \param elem  Data for tracking the dtype elements. If no element has been
 *              seen yet, its values are NULL. Note that an array may be
 * allocated,
 *              with zero-size arrays having some structure, before encountering
 *              an actual element.
 * \param arr The nd::array currently being populated. If a new type
 *            is encountered, this will be updated dynamically.
 * \param current_axis  The axis within shape being processed at
 *                      the current level of recursion.
 */
static void array_from_py_dynamic(PyObject *obj, std::vector<intptr_t> &shape,
                                  std::vector<afpd_coordentry> &coord,
                                  afpd_dtype &elem, dynd::nd::array &arr,
                                  intptr_t current_axis,
                                  const eval::eval_context *ectx)
{
  if (arr.is_null()) {
    // If the arr is NULL, we're doing the first recursion determining
    // number of dimensions, etc.
    array_from_py_dynamic_first_alloc(obj, shape, coord, elem, arr,
                                      current_axis, ectx);
    return;
  }

  // If it's the dtype, check for scalars
  if (current_axis == (intptr_t)shape.size()) {
    switch (elem.dtp.get_kind()) {
    case bool_kind:
      if (!bool_assign(coord[current_axis - 1].data_ptr, obj)) {
        promote_nd_arr_dtype(shape, coord, elem, arr,
                             deduce__type_from_pyobject(obj));
        array_broadcast_assign_from_py(elem.dtp, elem.arrmeta_ptr,
                                       coord[current_axis - 1].data_ptr, obj,
                                       ectx);
      }
      return;
    case sint_kind:
      if (!int_assign(elem.dtp, coord[current_axis - 1].data_ptr, obj)) {
        promote_nd_arr_dtype(shape, coord, elem, arr,
                             deduce__type_from_pyobject(obj));
        array_broadcast_assign_from_py(elem.dtp, elem.arrmeta_ptr,
                                       coord[current_axis - 1].data_ptr, obj,
                                       ectx);
      }
      return;
    case real_kind:
      if (!real_assign(coord[current_axis - 1].data_ptr, obj)) {
        promote_nd_arr_dtype(shape, coord, elem, arr,
                             deduce__type_from_pyobject(obj));
        array_broadcast_assign_from_py(elem.dtp, elem.arrmeta_ptr,
                                       coord[current_axis - 1].data_ptr, obj,
                                       ectx);
      }
      return;
    case complex_kind:
      if (!complex_assign(coord[current_axis - 1].data_ptr, obj)) {
        promote_nd_arr_dtype(shape, coord, elem, arr,
                             deduce__type_from_pyobject(obj));
        array_broadcast_assign_from_py(elem.dtp, elem.arrmeta_ptr,
                                       coord[current_axis - 1].data_ptr, obj,
                                       ectx);
      }
      return;
    case string_kind:
      if (!string_assign(elem.dtp, elem.arrmeta_ptr,
                         coord[current_axis - 1].data_ptr, obj, ectx)) {
        throw runtime_error("TODO: Handle string type promotion");
      }
      return;
#if PY_VERSION_HEX >= 0x03000000
    case bytes_kind:
      if (!bytes_assign(elem.dtp, elem.arrmeta_ptr,
                        coord[current_axis - 1].data_ptr, obj)) {
        throw runtime_error("TODO: Handle bytes type promotion");
      }
      return;
#endif
    case void_kind:
      // In this case, zero-sized dimension were encountered before
      // an actual element from which to deduce a type.
      promote_nd_arr_dtype(shape, coord, elem, arr,
                           deduce__type_from_pyobject(obj));
      array_broadcast_assign_from_py(elem.dtp, elem.arrmeta_ptr,
                                     coord[current_axis - 1].data_ptr, obj,
                                     ectx);
      return;
    default:
      throw runtime_error("internal error: unexpected type in recursive "
                          "pyobject to dynd array conversion");
    }
  }

  // Special case error for string and bytes, because they
  // support the sequence and iterator protocols, but we don't
  // want them to end up as dimensions.
  if (PyUnicode_Check(obj) ||
#if PY_VERSION_HEX >= 0x03000000
      PyBytes_Check(obj)) {
#else
      PyString_Check(obj)) {
#endif
    stringstream ss;
    ss << "Ragged dimension encountered, type ";
    pyobject_ownref typestr(PyObject_Str((PyObject *)Py_TYPE(obj)));
    ss << pystring_as_string(typestr.get());
    ss << " after a dimension type while converting to a dynd array";
    throw runtime_error(ss.str());
  }

  if (PyDict_Check(obj)) {
    throw invalid_argument("cannot automatically deduce dynd type to "
                           "convert dict into nd.array");
  }

  // We're processing a dimension
  if (shape[current_axis] >= 0) {
    // It's a strided dimension
    if (PySequence_Check(obj)) {
      Py_ssize_t size = PySequence_Size(obj);
      if (size != -1) {
        // The object supports the sequence protocol, so use it
        if (size != shape[current_axis]) {
          // Promote the current axis from strided to var
          promote_nd_arr_dim(shape, coord, elem, arr, current_axis, false);
          // Re-invoke this call, this time triggering the var dimension code
          array_from_py_dynamic(obj, shape, coord, elem, arr, current_axis,
                                ectx);
          return;
        }

        // In the strided case, the initial data pointer is the same
        // as the parent's. Note that for current_axis==0, arr.is_null()
        // is guaranteed to be true, so it is impossible to get here.
        coord[current_axis].data_ptr = coord[current_axis - 1].data_ptr;
        // Process all the elements
        for (intptr_t i = 0; i < size; ++i) {
          coord[current_axis].coord = i;
          pyobject_ownref item(PySequence_GetItem(obj, i));
          array_from_py_dynamic(item.get(), shape, coord, elem, arr,
                                current_axis + 1, ectx);
          // Advance to the next element. Because the array may be
          // dynamically reallocated deeper in the recursive call, we
          // need to get the stride from the arrmeta each time.
          const fixed_dim_type_arrmeta *md =
              reinterpret_cast<const fixed_dim_type_arrmeta *>(
                  coord[current_axis].arrmeta_ptr);
          coord[current_axis].data_ptr += md->stride;
        }
        return;
      } else {
        // If it doesn't actually check out as a sequence,
        // fall through to the iterator check.
        PyErr_Clear();
      }
    }

    // It wasn't a sequence, check if it's an iterator
    PyObject *iter = PyObject_GetIter(obj);
    if (iter != NULL) {
      pyobject_ownref iter_owner(iter);
      // In the strided case, the initial data pointer is the same
      // as the parent's. Note that for current_axis==0, arr.is_null()
      // is guaranteed to be true, so it is impossible to get here.
      coord[current_axis].data_ptr = coord[current_axis - 1].data_ptr;
      // Process all the elements
      Py_ssize_t size = shape[current_axis];
      for (intptr_t i = 0; i < size; ++i) {
        coord[current_axis].coord = i;
        PyObject *item = PyIter_Next(iter);
        if (item == NULL) {
          if (PyErr_Occurred()) {
            // Propagate any error
            throw exception();
          }
          // Promote the current axis from strided to var
          promote_nd_arr_dim(shape, coord, elem, arr, current_axis, true);
          // Shrink the var dim element to fit
          ndt::var_dim_element_resize(coord[current_axis].tp,
                                      coord[current_axis].arrmeta_ptr,
                                      coord[current_axis - 1].data_ptr, i);
          return;
        }
        pyobject_ownref item_ownref(item);
        array_from_py_dynamic(item, shape, coord, elem, arr, current_axis + 1,
                              ectx);
        // Advance to the next element. Because the array may be
        // dynamically reallocated deeper in the recursive call, we
        // need to get the stride from the arrmeta each time.
        const fixed_dim_type_arrmeta *md =
            reinterpret_cast<const fixed_dim_type_arrmeta *>(
                coord[current_axis].arrmeta_ptr);
        coord[current_axis].data_ptr += md->stride;
      }

      // Make sure the iterator has no more elements. If it does, we must
      // promote the dimension to var, then continue copying
      PyObject *item = PyIter_Next(iter);
      if (item != NULL) {
        pyobject_ownref item_ownref_outer(item);
        // Promote the current axis from strided to var
        promote_nd_arr_dim(shape, coord, elem, arr, current_axis, true);
        // Give the var dim some capacity so we can continue processing the
        // iterator
        coord[current_axis].reserved_size =
            std::max(size * 2, ARRAY_FROM_DYNAMIC_INITIAL_COUNT);
        ndt::var_dim_element_resize(coord[current_axis].tp,
                                    coord[current_axis].arrmeta_ptr,
                                    coord[current_axis - 1].data_ptr,
                                    coord[current_axis].reserved_size);
        item_ownref_outer.release();
        intptr_t i = size;
        while (item != NULL) {
          pyobject_ownref item_ownref(item);
          coord[current_axis].coord = i;
          char *data_ptr = coord[current_axis - 1].data_ptr;
          if (coord[current_axis].coord >= coord[current_axis].reserved_size) {
            // Increase the reserved capacity if needed
            coord[current_axis].reserved_size *= 2;
            ndt::var_dim_element_resize(
                coord[current_axis].tp, coord[current_axis].arrmeta_ptr,
                data_ptr, coord[current_axis].reserved_size);
          }
          // Set the data pointer for the child element
          var_dim_type_data *d =
              reinterpret_cast<var_dim_type_data *>(data_ptr);
          const var_dim_type_arrmeta *md =
              reinterpret_cast<const var_dim_type_arrmeta *>(
                  coord[current_axis].arrmeta_ptr);
          coord[current_axis].data_ptr = d->begin + i * md->stride;
          array_from_py_dynamic(item, shape, coord, elem, arr, current_axis + 1,
                                ectx);

          item = PyIter_Next(iter);
          ++i;
        }

        if (PyErr_Occurred()) {
          // Propagate any error
          throw exception();
        }
        // Shrink the var element to fit
        char *data_ptr = coord[current_axis - 1].data_ptr;
        ndt::var_dim_element_resize(coord[current_axis].tp,
                                    coord[current_axis].arrmeta_ptr, data_ptr,
                                    i);
        return;
      } else if (PyErr_Occurred()) {
        // Propagate any error
        throw exception();
      }
      return;
    }
  } else {
    // It's a var dimension
    if (PySequence_Check(obj)) {
      Py_ssize_t size = PySequence_Size(obj);
      if (size != -1) {
        ndt::var_dim_element_initialize(coord[current_axis].tp,
                                        coord[current_axis].arrmeta_ptr,
                                        coord[current_axis - 1].data_ptr, size);
        coord[current_axis].reserved_size = size;
        // Process all the elements
        for (intptr_t i = 0; i < size; ++i) {
          coord[current_axis].coord = i;
          pyobject_ownref item(PySequence_GetItem(obj, i));
          // Set the data pointer for the child element. We must
          // re-retrieve from `coord` each time, because the recursive
          // call could reallocate the destination array.
          var_dim_type_data *d = reinterpret_cast<var_dim_type_data *>(
              coord[current_axis - 1].data_ptr);
          const var_dim_type_arrmeta *md =
              reinterpret_cast<const var_dim_type_arrmeta *>(
                  coord[current_axis].arrmeta_ptr);
          coord[current_axis].data_ptr = d->begin + i * md->stride;
          array_from_py_dynamic(item, shape, coord, elem, arr, current_axis + 1,
                                ectx);
        }
        return;
      } else {
        // If it doesn't actually check out as a sequence,
        // fall through to the iterator check.
        PyErr_Clear();
      }
    }

    PyObject *iter = PyObject_GetIter(obj);
    if (iter != NULL) {
      pyobject_ownref iter_owner(iter);
      PyObject *item = PyIter_Next(iter);
      if (item != NULL) {
        pyobject_ownref item_ownref_outer(item);
        intptr_t i = 0;
        coord[current_axis].reserved_size = ARRAY_FROM_DYNAMIC_INITIAL_COUNT;
        ndt::var_dim_element_initialize(coord[current_axis].tp,
                                        coord[current_axis].arrmeta_ptr,
                                        coord[current_axis - 1].data_ptr,
                                        coord[current_axis].reserved_size);
        item_ownref_outer.release();
        while (item != NULL) {
          pyobject_ownref item_ownref(item);
          coord[current_axis].coord = i;
          char *data_ptr = coord[current_axis - 1].data_ptr;
          if (coord[current_axis].coord >= coord[current_axis].reserved_size) {
            // Increase the reserved capacity if needed
            coord[current_axis].reserved_size *= 2;
            ndt::var_dim_element_resize(
                coord[current_axis].tp, coord[current_axis].arrmeta_ptr,
                data_ptr, coord[current_axis].reserved_size);
          }
          // Set the data pointer for the child element
          var_dim_type_data *d =
              reinterpret_cast<var_dim_type_data *>(data_ptr);
          const var_dim_type_arrmeta *md =
              reinterpret_cast<const var_dim_type_arrmeta *>(
                  coord[current_axis].arrmeta_ptr);
          coord[current_axis].data_ptr = d->begin + i * md->stride;
          array_from_py_dynamic(item, shape, coord, elem, arr, current_axis + 1,
                                ectx);

          item = PyIter_Next(iter);
          ++i;
        }

        if (PyErr_Occurred()) {
          // Propagate any error
          throw exception();
        }
        // Shrink the var element to fit
        char *data_ptr = coord[current_axis - 1].data_ptr;
        ndt::var_dim_element_resize(coord[current_axis].tp,
                                    coord[current_axis].arrmeta_ptr, data_ptr,
                                    i);
        return;
      } else {
        // Set it to a zero-sized var element
        char *data_ptr = coord[current_axis - 1].data_ptr;
        ndt::var_dim_element_initialize(coord[current_axis].tp,
                                        coord[current_axis].arrmeta_ptr,
                                        data_ptr, 0);
        return;
      }
    } else {
      if (PyErr_ExceptionMatches(PyExc_TypeError)) {
        // A TypeError indicates that the object doesn't support
        // the iterator protocol
        PyErr_Clear();
      } else {
        // Propagate the error
        throw exception();
      }
    }
  }

  // The object supported neither the sequence nor the iterator
  // protocol, so report an error.
  stringstream ss;
  ss << "Ragged dimension encountered, type ";
  pyobject_ownref typestr(PyObject_Str((PyObject *)Py_TYPE(obj)));
  ss << pystring_as_string(typestr.get());
  ss << " after a dimension type while converting to a dynd array";
  throw runtime_error(ss.str());
}

dynd::nd::array pydynd::array_from_py_dynamic(PyObject *obj,
                                              const eval::eval_context *ectx)
{
  std::vector<afpd_coordentry> coord;
  std::vector<intptr_t> shape;
  afpd_dtype elem;
  nd::array arr;
  memset(&elem, 0, sizeof(elem));
  array_from_py_dynamic(obj, shape, coord, elem, arr, 0, ectx);
  // Finalize any variable-sized buffers, etc
  if (!arr.get_type().is_builtin()) {
    arr.get_type().extended()->arrmeta_finalize_buffers(arr.get()->metadata());
  }
  // As a special case, convert the outer dimension from var to strided
  if (arr.get_type().get_type_id() == var_dim_type_id) {
    arr = arr.view(ndt::make_fixed_dim(
        arr.get_dim_size(),
        arr.get_type().extended<ndt::base_dim_type>()->get_element_type()));
  }
  return arr;
}

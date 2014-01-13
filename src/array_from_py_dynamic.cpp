//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>
#include <datetime.h>

#include <dynd/types/string_type.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/base_struct_type.hpp>
#include <dynd/types/date_type.hpp>
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

// Initialize the pydatetime API
namespace {
struct init_pydatetime {
    init_pydatetime() {
        PyDateTime_IMPORT;
    }
};
init_pydatetime pdt;

struct afpd_coordentry {
    // The current coordinate of this axis being processed
    intptr_t coord;
    // The type in the output array for this axis
    ndt::type tp;
    // The metadata pointer in the output array for this axis
    const char *metadata_ptr;
    // The data pointer in the output array for the next axis (or element)
    char *data_ptr;
    // Used for var dimensions, the amount of presently reserved space
    intptr_t reserved_size;
};

struct afpd_dtype {
    // The data type after all the dimensions
    ndt::type dtp;
    // The metadata pointer in the output array for the dtype
    const char *metadata_ptr;

    void swap(afpd_dtype& rhs) {
        dtp.swap(rhs.dtp);
        std::swap(metadata_ptr, rhs.metadata_ptr);
    }
};

} // anonymous namespace

static void array_from_py_dynamic(
    PyObject *obj,
    std::vector<intptr_t>& shape,
    std::vector<afpd_coordentry>& coord,
    afpd_dtype& elem,
    dynd::nd::array& arr,
    intptr_t current_axis);

/**
 * This allocates an nd::array for the first time,
 * using the shape provided, and filling in the
 * `coord` and `elem`.
 */
static nd::array allocate_nd_arr(
    std::vector<intptr_t>& shape,
    std::vector<afpd_coordentry>& coord,
    afpd_dtype& elem)
{
    intptr_t ndim = (intptr_t)shape.size();
    // Allocate the nd::array
    nd::array result = nd::make_strided_array(
                elem.dtp, ndim,
                ndim == 0 ? NULL : &shape[0]);
    // Fill `coord` with pointers from the allocated arrays,
    // reserving some data for any var dimensions.
    coord.resize(shape.size());
    ndt::type tp = result.get_type();
    const char *metadata_ptr = result.get_ndo_meta();
    char *data_ptr = result.get_readwrite_originptr();
    for (intptr_t i = 0; i < ndim; ++i) {
        afpd_coordentry& c = coord[i];
        c.coord = 0;
        c.tp = tp;
        c.metadata_ptr = metadata_ptr;
        // If it's a var dim, reserve some space
        if (tp.get_type_id() == var_dim_type_id) {
            intptr_t initial_count = ARRAY_FROM_DYNAMIC_INITIAL_COUNT;
            ndt::var_dim_element_initialize(tp, metadata_ptr,
                                data_ptr, initial_count);
            c.reserved_size = initial_count;
            // Advance metadata_ptr and data_ptr to the child dimension
            metadata_ptr += sizeof(var_dim_type_metadata);
            data_ptr = reinterpret_cast<const var_dim_type_data *>(data_ptr)->begin;
            tp = static_cast<const var_dim_type *>(tp.extended())->get_element_type();
        } else {
            // Advance metadata_ptr and data_ptr to the child dimension
            metadata_ptr += sizeof(strided_dim_type_metadata);
            tp = static_cast<const strided_dim_type *>(tp.extended())->get_element_type();
        }
        c.data_ptr = data_ptr;
    }
    elem.metadata_ptr = metadata_ptr;

    return result;
}

/**
 * This copies everything up to, but not including, the
 * current coordinate in `src_coord`. When finished,
 * `dst_coord` is left in a state equivalent to `src_coord`.
 *
 * This is for the case where the dtype was promoted to
 * a new type.
 */
static void copy_to_promoted_nd_arr(
    std::vector<intptr_t>& shape,
    char *dst_data_ptr,
    std::vector<afpd_coordentry>& dst_coord,
    afpd_dtype& dst_elem,
    const char *src_data_ptr,
    std::vector<afpd_coordentry>& src_coord,
    afpd_dtype& src_elem,
    const assignment_strided_ckernel_builder& ck,
    intptr_t current_axis,
    bool final_coordinate)
{
    intptr_t ndim = shape.size();
    if (current_axis == ndim - 1) {
        // Base case - the final dimension
        if (shape[current_axis] >= 0) {
            // strided dimension case
            const strided_dim_type_metadata *dst_md =
                reinterpret_cast<const strided_dim_type_metadata *>(dst_coord[current_axis].metadata_ptr);
            const strided_dim_type_metadata *src_md =
                reinterpret_cast<const strided_dim_type_metadata *>(src_coord[current_axis].metadata_ptr);
            if (!final_coordinate) {
                // Copy the full dimension
                ck(dst_data_ptr, dst_md->stride,
                    src_data_ptr, src_md->stride, shape[current_axis]);
            } else {
                // Copy up to, but not including, the coordinate
                ck(dst_data_ptr, dst_md->stride,
                    src_data_ptr, src_md->stride, src_coord[current_axis].coord);
                dst_coord[current_axis].coord = src_coord[current_axis].coord;
                dst_coord[current_axis].data_ptr = dst_data_ptr + dst_md->stride * dst_coord[current_axis].coord;
            }
        } else {
            // var dimension case
            const var_dim_type_metadata *dst_md =
                reinterpret_cast<const var_dim_type_metadata *>(dst_coord[current_axis].metadata_ptr);
            const var_dim_type_metadata *src_md =
                reinterpret_cast<const var_dim_type_metadata *>(src_coord[current_axis].metadata_ptr);
            var_dim_type_data *dst_d =
                reinterpret_cast<var_dim_type_data *>(dst_data_ptr);
            const var_dim_type_data *src_d =
                reinterpret_cast<const var_dim_type_data *>(src_data_ptr);
            if (!final_coordinate) {
                ndt::var_dim_element_resize(dst_coord[current_axis].tp,
                            dst_coord[current_axis].metadata_ptr,
                            dst_data_ptr, src_d->size);
                // Copy the full dimension
                ck(dst_d->begin, dst_md->stride,
                    src_d->begin, src_md->stride, src_d->size);
            } else {
                // Initialize the var element to the same reserved space as the input
                ndt::var_dim_element_resize(dst_coord[current_axis].tp,
                            dst_coord[current_axis].metadata_ptr,
                            dst_data_ptr, src_coord[current_axis].reserved_size);
                dst_coord[current_axis].reserved_size = src_coord[current_axis].reserved_size;
                // Copy up to, but not including, the coordinate
                ck(dst_d->begin, dst_md->stride,
                    src_d->begin, src_md->stride, src_coord[current_axis].coord);
                dst_coord[current_axis].coord = src_coord[current_axis].coord;
                dst_coord[current_axis].data_ptr = dst_d->begin +
                            dst_md->stride * dst_coord[current_axis].coord;
            }
        }
    } else {
        // Recursive case
        if (shape[current_axis] >= 0) {
            // strided dimension case
            const strided_dim_type_metadata *dst_md =
                reinterpret_cast<const strided_dim_type_metadata *>(dst_coord[current_axis].metadata_ptr);
            const strided_dim_type_metadata *src_md =
                reinterpret_cast<const strided_dim_type_metadata *>(src_coord[current_axis].metadata_ptr);
            if (!final_coordinate) {
                // Copy the full dimension
                intptr_t size = shape[current_axis];
                intptr_t dst_stride = dst_md->stride;
                intptr_t src_stride = src_md->stride;
                for (intptr_t i = 0; i < size; ++i,
                                               dst_data_ptr += dst_stride,
                                               src_data_ptr += src_stride) {
                    copy_to_promoted_nd_arr(shape,
                        dst_data_ptr, dst_coord, dst_elem,
                        src_data_ptr, src_coord, src_elem,
                        ck, current_axis + 1, false);
                }
            } else {
                // Copy up to, and including, the coordinate
                intptr_t size = src_coord[current_axis].coord;
                intptr_t dst_stride = dst_md->stride;
                intptr_t src_stride = src_md->stride;
                dst_coord[current_axis].coord = size;
                dst_coord[current_axis].data_ptr = dst_data_ptr + dst_stride * size;
                for (intptr_t i = 0; i <= size; ++i,
                                               dst_data_ptr += dst_stride,
                                               src_data_ptr += src_stride) {
                    copy_to_promoted_nd_arr(shape,
                        dst_data_ptr, dst_coord, dst_elem,
                        src_data_ptr, src_coord, src_elem,
                        ck, current_axis + 1, i == size);
                }
            }
        } else {
            // var dimension case
            const var_dim_type_metadata *dst_md =
                reinterpret_cast<const var_dim_type_metadata *>(dst_coord[current_axis].metadata_ptr);
            const var_dim_type_metadata *src_md =
                reinterpret_cast<const var_dim_type_metadata *>(src_coord[current_axis].metadata_ptr);
            var_dim_type_data *dst_d =
                reinterpret_cast<var_dim_type_data *>(dst_data_ptr);
            const var_dim_type_data *src_d =
                reinterpret_cast<const var_dim_type_data *>(src_data_ptr);
            if (!final_coordinate) {
                ndt::var_dim_element_resize(dst_coord[current_axis].tp,
                            dst_coord[current_axis].metadata_ptr,
                            dst_data_ptr, src_d->size);
                // Copy the full dimension
                intptr_t size = src_d->size;
                char *dst_elem_ptr = dst_d->begin;
                intptr_t dst_stride = dst_md->stride;
                const char *src_elem_ptr = src_d->begin;
                intptr_t src_stride = src_md->stride;
                for (intptr_t i = 0; i < size; ++i,
                                               dst_elem_ptr += dst_stride,
                                               src_elem_ptr += src_stride) {
                    copy_to_promoted_nd_arr(shape,
                        dst_elem_ptr, dst_coord, dst_elem,
                        src_elem_ptr, src_coord, src_elem,
                        ck, current_axis + 1, false);
                }
            } else {
                // Initialize the var element to the same reserved space as the input
                ndt::var_dim_element_resize(dst_coord[current_axis].tp,
                            dst_coord[current_axis].metadata_ptr,
                            dst_data_ptr, src_coord[current_axis].reserved_size);
                dst_coord[current_axis].reserved_size = src_coord[current_axis].reserved_size;
                // Copy up to, and including, the size
                intptr_t size = src_coord[current_axis].coord;
                char *dst_elem_ptr = dst_d->begin;
                intptr_t dst_stride = dst_md->stride;
                const char *src_elem_ptr = src_d->begin;
                intptr_t src_stride = src_md->stride;
                dst_coord[current_axis].coord = size;
                dst_coord[current_axis].data_ptr = dst_elem_ptr + dst_stride * size;
                for (intptr_t i = 0; i <= size; ++i,
                                               dst_elem_ptr += dst_stride,
                                               src_elem_ptr += src_stride) {
                    copy_to_promoted_nd_arr(shape,
                        dst_elem_ptr, dst_coord, dst_elem,
                        src_elem_ptr, src_coord, src_elem,
                        ck, current_axis + 1, i == size);
                }
            }
        }
    }
}

/**
 * This function promotes the dtype the array currently has with
 * `tp`, allocates a new one, then copies all the data up to the
 * current index in `coord`.
 */
static void promote_nd_arr(
    std::vector<intptr_t>& shape,
    std::vector<afpd_coordentry>& coord,
    afpd_dtype& elem,
    nd::array& arr,
    const ndt::type& tp)
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
    nd::array newarr = allocate_nd_arr(shape, newcoord, newelem);
    // Copy the data up to, but not including, the current `coord`
    // from the old `arr` to the new one
    assignment_strided_ckernel_builder k;
    if (elem.dtp.get_type_id() != uninitialized_type_id) {
        make_assignment_kernel(&k, 0, newelem.dtp, newelem.metadata_ptr,
                        elem.dtp, elem.metadata_ptr,
                        kernel_request_strided,
                        assign_error_none, &eval::default_eval_context);
    }
    copy_to_promoted_nd_arr(shape, newarr.get_readwrite_originptr(),
                newcoord, newelem, arr.get_readonly_originptr(),
                coord, elem, k, 0, true);
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

static bool int_assign(const ndt::type& tp, char *data, PyObject *obj)
{
#if PY_VERSION_HEX < 0x03000000
    if (PyInt_Check(obj)) {
        long value = PyInt_AS_LONG(obj);
# if SIZEOF_LONG > SIZEOF_INT
        // Check whether we're assigning to int32 or int64
        if (tp.get_type_id() == int64_type_id) {
            int64_t *result_ptr = reinterpret_cast<int64_t *>(data);
            *result_ptr = static_cast<int64_t>(value);
        } else {
            if (value >= INT_MIN && value <= INT_MAX) {
                int32_t *result_ptr = reinterpret_cast<int32_t *>(data);
                *result_ptr = static_cast<int32_t>(value);
            } else {
                // Needs promotion to int64
                return false;
            }
        }
# else
        int32_t *result_ptr = reinterpret_cast<int32_t *>(data);
        *result_ptr = static_cast<int32_t>(value);
# endif
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
    complex<double> *result_ptr = reinterpret_cast<complex<double> *>(data);
    if (PyComplex_Check(obj)) {
        *result_ptr = complex<double>(PyComplex_RealAsDouble(obj),
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
 * \param  metadata  Metadata for the string.
 * \param  data  The data element being assigned to.
 * \param  obj  The Python object containing the value
 *
 * \return  True if the assignment was successful, false if the input type is incompatible.
 */
static bool string_assign(const ndt::type& tp, const char *metadata, char *data, PyObject *obj)
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

        const string_type *st = static_cast<const string_type *>(tp.extended());
        st->set_utf8_string(metadata, data, assign_error_default, s, s + len);
        return true;
    }
#if PY_VERSION_HEX < 0x03000000
    else if (PyString_Check(obj)) {
        char *s = NULL;
        intptr_t len = 0;
        if (PyString_AsStringAndSize(obj, &s, &len) < 0) {
            throw runtime_error("Error getting string data");
        }

        const string_type *st = static_cast<const string_type *>(tp.extended());
        st->set_utf8_string(metadata, data, assign_error_default, s, s + len);
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
 * \param  metadata  Metadata for the type.
 * \param  data  The data element being assigned to.
 * \param  obj  The Python object containing the value
 *
 * \return  True if the assignment was successful, false if the input type is incompatible.
 */
static bool bytes_assign(const ndt::type& tp, const char *metadata, char *data, PyObject *obj)
{
    if (PyBytes_Check(obj)) {
        char *s = NULL;
        intptr_t len = 0;
        if (PyBytes_AsStringAndSize(obj, &s, &len) < 0) {
            throw runtime_error("Error getting bytes data");
        }

        const bytes_type *st = static_cast<const bytes_type *>(tp.extended());
        st->set_bytes_data(metadata, data, s, s + len);
        return true;
    }
    else {
        return false;
    }
}
#endif

static void array_from_py_dynamic_first_alloc(
    PyObject *obj,
    std::vector<intptr_t>& shape,
    std::vector<afpd_coordentry>& coord,
    afpd_dtype& elem,
    dynd::nd::array& arr,
    intptr_t current_axis)
{
    if (PyUnicode_Check(obj)
#if PY_VERSION_HEX < 0x03000000
                    || PyString_Check(obj)
#endif
                    ) {
        // Special case strings, because they act as sequences too
        elem.dtp = ndt::make_string();
        arr = allocate_nd_arr(shape, coord, elem);
        string_assign(elem.dtp, elem.metadata_ptr, coord[current_axis-1].data_ptr, obj);
        return;
    }

#if PY_VERSION_HEX >= 0x03000000
    if (PyBytes_Check(obj)) {
        // Special case bytes, because they act as sequences too
        elem.dtp = ndt::make_bytes(1);
        arr = allocate_nd_arr(shape, coord, elem);
        bytes_assign(elem.dtp, elem.metadata_ptr, coord[current_axis-1].data_ptr, obj);
        return;
    }
#endif

    if (PySequence_Check(obj)) {
        Py_ssize_t size = PySequence_Size(obj);
        if (size != -1) {
            // Add this size to the shape
            shape.push_back(size);
            // Initialize the data pointer for child elements with the one for this element
            if (!coord.empty() && current_axis > 0) {
                coord[current_axis].data_ptr = coord[current_axis-1].data_ptr;
            }
            // Process all the elements
            for (intptr_t i = 0; i < size; ++i) {
                if (!coord.empty()) {
                    coord[current_axis].coord = i;
                }
                pyobject_ownref item(PySequence_GetItem(obj, i));
                array_from_py_dynamic(item.get(), shape, coord, elem,
                            arr, current_axis + 1);
                // Advance to the next element. Because the array may be
                // dynamically reallocated deeper in the recursive call, we
                // need to get the stride from the metadata each time.
                const strided_dim_type_metadata *md =
                    reinterpret_cast<const strided_dim_type_metadata *>(coord[current_axis].metadata_ptr);
                coord[current_axis].data_ptr += md->stride;
            }
            return;
        } else {
			// If it doesn't actually check out as a sequence,
			// fall through eventually to the iterator check.
            PyErr_Clear();
        }
    }

    if (PyBool_Check(obj)) {
        elem.dtp = ndt::make_type<dynd_bool>();
        arr = allocate_nd_arr(shape, coord, elem);
        *coord[current_axis-1].data_ptr = (obj == Py_True);
        return;
    }

#if PY_VERSION_HEX < 0x03000000
    if (PyInt_Check(obj)) {
        long value = PyInt_AS_LONG(obj);
# if SIZEOF_LONG > SIZEOF_INT
        // Use a 32-bit int if it fits.
        if (value >= INT_MIN && value <= INT_MAX) {
            elem.dtp = ndt::make_type<int32_t>();
            arr = allocate_nd_arr(shape, coord, elem);
            int32_t *result_ptr = reinterpret_cast<int32_t *>(coord[current_axis-1].data_ptr);
            *result_ptr = static_cast<int32_t>(value);
        } else {
            elem.dtp = ndt::make_type<int64_t>();
            arr = allocate_nd_arr(shape, coord, elem);
            int64_t *result_ptr = reinterpret_cast<int64_t *>(coord[current_axis-1].data_ptr);
            *result_ptr = static_cast<int64_t>(value);
        }
# else
        elem.dtp = ndt::make_type<int32_t>();
        arr = allocate_nd_arr(shape, coord, elem);
        int32_t *result_ptr = reinterpret_cast<int32_t *>(coord[current_axis-1].data_ptr);
        *result_ptr = static_cast<int32_t>(value);
# endif
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
            elem.dtp = ndt::make_type<int32_t>();
            arr = allocate_nd_arr(shape, coord, elem);
            int32_t *result_ptr = reinterpret_cast<int32_t *>(coord[current_axis-1].data_ptr);
            *result_ptr = static_cast<int32_t>(value);
        } else {
            elem.dtp = ndt::make_type<int64_t>();
            arr = allocate_nd_arr(shape, coord, elem);
            int64_t *result_ptr = reinterpret_cast<int64_t *>(coord[current_axis-1].data_ptr);
            *result_ptr = static_cast<int64_t>(value);
        }
        return;
    }

    if (PyFloat_Check(obj)) {
        elem.dtp = ndt::make_type<double>();
        arr = allocate_nd_arr(shape, coord, elem);
        double *result_ptr = reinterpret_cast<double *>(coord[current_axis-1].data_ptr);
        *result_ptr = PyFloat_AS_DOUBLE(obj);
        return;
    }

    if (PyComplex_Check(obj)) {
        elem.dtp = ndt::make_type<complex<double> >();
        arr = allocate_nd_arr(shape, coord, elem);
        complex<double> *result_ptr = reinterpret_cast<complex<double> *>(coord[current_axis-1].data_ptr);
        *result_ptr = complex<double>(PyComplex_RealAsDouble(obj),
                                PyComplex_ImagAsDouble(obj));
        return;
    }

    // Check if it's an iterator
    {
        PyObject *iter = PyObject_GetIter(obj);
        if (iter != NULL) {
            // Indicate a var dim.
            shape.push_back(-1);
            PyObject *item = PyIter_Next(iter);
            if (item != NULL) {
                intptr_t i = 0;
                while (item != NULL) {
                    pyobject_ownref item_ownref(item);
                    if (!coord.empty()) {
                        coord[current_axis].coord = i;
                        char *data_ptr = (current_axis > 0) ? coord[current_axis-1].data_ptr
                                                            : arr.get_readwrite_originptr();
                        if (coord[current_axis].coord >= coord[current_axis].reserved_size) {
                            // Increase the reserved capacity if needed
                            coord[current_axis].reserved_size *= 2;
                            ndt::var_dim_element_resize(coord[current_axis].tp,
                                        coord[current_axis].metadata_ptr,
                                        data_ptr, coord[current_axis].reserved_size);
                        }
                        // Set the data pointer for the child element
                        var_dim_type_data *d = reinterpret_cast<var_dim_type_data *>(data_ptr);
                        const var_dim_type_metadata *md =
                            reinterpret_cast<const var_dim_type_metadata *>(coord[current_axis].metadata_ptr);
                        coord[current_axis].data_ptr = d->begin + i * md->stride;
                    }
                    array_from_py_dynamic(item, shape, coord, elem,
                                arr, current_axis + 1);

                    item = PyIter_Next(iter);
                    ++i;
                }

                if (PyErr_Occurred()) {
                    // Propagate any error
                    throw exception();
                }
                // Shrink the var element to fit
                char *data_ptr = (current_axis > 0) ? coord[current_axis-1].data_ptr
                                                    : arr.get_readwrite_originptr();
                ndt::var_dim_element_resize(coord[current_axis].tp,
                            coord[current_axis].metadata_ptr,
                            data_ptr, i);
                return;
            } else {
                // Because this iterator's sequence is zero-sized, we can't
                // start deducing the type yet. To start capturing the
                // dimensional structure, we make an array of
                // int32, while keeping elem.dtp as an uninitialized type.
                elem.dtp = ndt::make_type<int32_t>();
                arr = allocate_nd_arr(shape, coord, elem);
                // Make the dtype uninitialized again, to signal we have
                // deduced anything yet.
                elem.dtp = ndt::type();
                // Set it to a zero-sized var element
                char *data_ptr = (current_axis > 0) ? coord[current_axis-1].data_ptr
                                                    : arr.get_readwrite_originptr();
                ndt::var_dim_element_resize(coord[current_axis].tp,
                            coord[current_axis].metadata_ptr, data_ptr, 0);
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
 *              seen yet, its values are NULL. Note that an array may be allocated,
 *              with zero-size arrays having some structure, before encountering
 *              an actual element.
 * \param arr The nd::array currently being populated. If a new type
 *            is encountered, this will be updated dynamically.
 * \param current_axis  The axis within shape being processed at
 *                      the current level of recursion.
 */
static void array_from_py_dynamic(
    PyObject *obj,
    std::vector<intptr_t>& shape,
    std::vector<afpd_coordentry>& coord,
    afpd_dtype& elem,
    dynd::nd::array& arr,
    intptr_t current_axis)
{
    if (arr.is_empty()) {
        // If the arr is NULL, we're doing the first recursion determining
        // number of dimensions, etc.
        array_from_py_dynamic_first_alloc(obj, shape, coord, elem, arr, current_axis);
        return;
    }

    // If it's the dtype, check for scalars
    if (current_axis == shape.size()) {
        switch (elem.dtp.get_kind()) {
            case bool_kind:
                if (!bool_assign(coord[current_axis-1].data_ptr, obj)) {
                    promote_nd_arr(shape, coord, elem, arr,
                        promote_types_arithmetic(elem.dtp, deduce_ndt_type_from_pyobject(obj)));
                    array_broadcast_assign_from_py(elem.dtp, elem.metadata_ptr,
                                coord[current_axis-1].data_ptr, obj);
                }
                return;
            case int_kind:
                if (!int_assign(elem.dtp, coord[current_axis-1].data_ptr, obj)) {
                    promote_nd_arr(shape, coord, elem, arr,
                        promote_types_arithmetic(elem.dtp, deduce_ndt_type_from_pyobject(obj)));
                    array_broadcast_assign_from_py(elem.dtp, elem.metadata_ptr,
                                coord[current_axis-1].data_ptr, obj);
                }
                return;
            case real_kind:
                if (!real_assign(coord[current_axis-1].data_ptr, obj)) {
                    promote_nd_arr(shape, coord, elem, arr,
                        promote_types_arithmetic(elem.dtp, deduce_ndt_type_from_pyobject(obj)));
                    array_broadcast_assign_from_py(elem.dtp, elem.metadata_ptr,
                                coord[current_axis-1].data_ptr, obj);
                }
                return;
            case complex_kind:
                if (!complex_assign(coord[current_axis-1].data_ptr, obj)) {
                    promote_nd_arr(shape, coord, elem, arr,
                        promote_types_arithmetic(elem.dtp, deduce_ndt_type_from_pyobject(obj)));
                    array_broadcast_assign_from_py(elem.dtp, elem.metadata_ptr,
                                coord[current_axis-1].data_ptr, obj);
                }
                return;
            case string_kind:
                if (!string_assign(elem.dtp, elem.metadata_ptr,
                                coord[current_axis-1].data_ptr, obj)) {
                    throw runtime_error("TODO: Handle string type promotion");
                }
                return;
#if PY_VERSION_HEX >= 0x03000000
            case bytes_kind:
                if (!bytes_assign(elem.dtp, elem.metadata_ptr,
                                coord[current_axis-1].data_ptr, obj)) {
                    throw runtime_error("TODO: Handle bytes type promotion");
                }
                return;
#endif
            case void_kind:
                // In this case, zero-sized dimension were encountered before
                // an actual element from which to deduce a type.
                throw runtime_error("TODO: Handle late discovery of dtype in dynd conversion");
            default:
                throw runtime_error("internal error: unexpected type in recursive pyobject to dynd array conversion");
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

    // We're processing a dimension
    if (shape[current_axis] >= 0) {
        // It's a strided dimension
        if (PySequence_Check(obj)) {
            Py_ssize_t size = PySequence_Size(obj);
            if (size != -1) {
                // The object supports the sequence protocol, so use it
                if (size != shape[current_axis]) {
                    throw runtime_error("TODO: promote from strided to var dimension");
                }

                // In the strided case, the initial data pointer is the same
                // as the parent's. Note that for current_axis==0, arr.is_empty()
                // is guaranteed to be true, so it is impossible to get here.
                coord[current_axis].data_ptr = coord[current_axis-1].data_ptr;
                // Process all the elements
                for (intptr_t i = 0; i < size; ++i) {
                    coord[current_axis].coord = i;
                    pyobject_ownref item(PySequence_GetItem(obj, i));
                    array_from_py_dynamic(item.get(), shape, coord, elem,
                                arr, current_axis + 1);
                    // Advance to the next element. Because the array may be
                    // dynamically reallocated deeper in the recursive call, we
                    // need to get the stride from the metadata each time.
                    const strided_dim_type_metadata *md =
                        reinterpret_cast<const strided_dim_type_metadata *>(coord[current_axis].metadata_ptr);
                    coord[current_axis].data_ptr += md->stride;
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
            Py_DECREF(iter);
            throw runtime_error("TODO: handle getting an iterator for a strided array");
        }
    } else {
        // It's a var dimension
        if (PySequence_Check(obj)) {
            Py_ssize_t size = PySequence_Size(obj);
            if (size != -1) {
                ndt::var_dim_element_initialize(coord[current_axis].tp,
                            coord[current_axis].metadata_ptr,
                            coord[current_axis-1].data_ptr, size);
                coord[current_axis].reserved_size = size;
                // Process all the elements
                for (intptr_t i = 0; i < size; ++i) {
                    coord[current_axis].coord = i;
                    pyobject_ownref item(PySequence_GetItem(obj, i));
                    // Set the data pointer for the child element. We must
                    // re-retrieve from `coord` each time, because the recursive
                    // call could reallocate the destination array.
                    var_dim_type_data *d = reinterpret_cast<var_dim_type_data *>(coord[current_axis-1].data_ptr);
                    const var_dim_type_metadata *md =
                        reinterpret_cast<const var_dim_type_metadata *>(coord[current_axis].metadata_ptr);
                    coord[current_axis].data_ptr = d->begin + i * md->stride;
                    array_from_py_dynamic(item, shape, coord, elem,
                                arr, current_axis + 1);
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
            PyObject *item = PyIter_Next(iter);
            if (item != NULL) {
                intptr_t i = 0;
                coord[current_axis].reserved_size = ARRAY_FROM_DYNAMIC_INITIAL_COUNT;
                ndt::var_dim_element_initialize(coord[current_axis].tp,
                            coord[current_axis].metadata_ptr,
                            coord[current_axis-1].data_ptr,
                            coord[current_axis].reserved_size);
                while (item != NULL) {
                    pyobject_ownref item_ownref(item);
                    coord[current_axis].coord = i;
                    char *data_ptr = coord[current_axis-1].data_ptr;
                    if (coord[current_axis].coord >= coord[current_axis].reserved_size) {
                        // Increase the reserved capacity if needed
                        coord[current_axis].reserved_size *= 2;
                        ndt::var_dim_element_resize(coord[current_axis].tp,
                                    coord[current_axis].metadata_ptr,
                                    data_ptr, coord[current_axis].reserved_size);
                    }
                    // Set the data pointer for the child element
                    var_dim_type_data *d = reinterpret_cast<var_dim_type_data *>(data_ptr);
                    const var_dim_type_metadata *md =
                        reinterpret_cast<const var_dim_type_metadata *>(coord[current_axis].metadata_ptr);
                    coord[current_axis].data_ptr = d->begin + i * md->stride;
                    array_from_py_dynamic(item, shape, coord, elem,
                                arr, current_axis + 1);

                    item = PyIter_Next(iter);
                    ++i;
                }

                if (PyErr_Occurred()) {
                    // Propagate any error
                    throw exception();
                }
                // Shrink the var element to fit
                char *data_ptr = coord[current_axis-1].data_ptr;
                ndt::var_dim_element_resize(coord[current_axis].tp,
                            coord[current_axis].metadata_ptr,
                            data_ptr, i);
                return;
            } else {
                // Set it to a zero-sized var element
                char *data_ptr = coord[current_axis-1].data_ptr;
                ndt::var_dim_element_initialize(coord[current_axis].tp,
                            coord[current_axis].metadata_ptr, data_ptr, 0);
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

dynd::nd::array pydynd::array_from_py_dynamic(PyObject *obj)
{
    std::vector<afpd_coordentry> coord;
    std::vector<intptr_t> shape;
    afpd_dtype elem;
    nd::array arr;
    memset(&elem, 0, sizeof(elem));
    array_from_py_dynamic(obj, shape, coord, elem, arr, 0);
    return arr;
}

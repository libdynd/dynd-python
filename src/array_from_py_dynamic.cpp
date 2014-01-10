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
#include <dynd/memblock/external_memory_block.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/type_promotion.hpp>
#include <dynd/exceptions.hpp>

#include "array_from_py.hpp"
#include "array_from_py_dynamic.hpp"
#include "array_assign_from_py.hpp"
#include "array_functions.hpp"
#include "type_functions.hpp"
#include "utility_functions.hpp"
#include "numpy_interop.hpp"

using namespace std;
using namespace dynd;
using namespace pydynd;

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
inline nd::array allocate_nd_arr(
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
            intptr_t initial_count = 16;
            ndt::var_dim_element_initialize(tp, metadata_ptr,
                                data_ptr, initial_count);
            c.reserved_size = initial_count;
            // Advance metadata_ptr and data_ptr to the child dimension
            metadata_ptr += sizeof(var_dim_type_metadata);
            data_ptr = reinterpret_cast<const var_dim_type_data *>(data_ptr)->begin;
        } else {
            // Advance metadata_ptr and data_ptr to the child dimension
            metadata_ptr += sizeof(strided_dim_type_metadata);
        }
        c.data_ptr = data_ptr;
    }
    elem.metadata_ptr = metadata_ptr;

    return result;
}

/**
 * Special string assignment, just for UTF-8 because
 * that's the type of string we produce.
 */
static void string_assign(const ndt::type& tp, const char *metadata, char *data, PyObject *obj)
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
    }
#if PY_VERSION_HEX < 0x03000000
    else if (PyString_Check(obj)) {
        char *pystr_data = NULL;
        intptr_t pystr_len = 0;
        if (PyString_AsStringAndSize(obj, &pystr_data, &pystr_len) < 0) {
            throw runtime_error("Error getting string data");
        }

        const string_type *st = static_cast<const string_type *>(tp.extended());
        st->set_utf8_string(metadata, data, assign_error_default, s, s + len);
    }
#endif
    else {
        throw runtime_error("internal error: string assignment should get a String or Unicode");
    }
}

#if PY_VERSION_HEX >= 0x03000000
static void bytes_assign(const ndt::type& tp, const char *metadata, char *data, PyObject *obj)
{
    if (PyBytes_Check(obj)) {
        char *s = NULL;
        intptr_t len = 0;
        if (PyBytes_AsStringAndSize(obj, &s, &len) < 0) {
            throw runtime_error("Error getting bytes data");
        }

        const bytes_type *st = static_cast<const bytes_type *>(tp.extended());
        st->set_bytes_data(metadata, data, assign_error_default, s, s + len);
    }
    else {
        throw runtime_error("internal error: bytes assignment should get a Bytes");
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
            *result_ptr = static_cast<int>(value);
        } else {
            elem.dtp = ndt::make_type<int64_t>();
            arr = allocate_nd_arr(shape, coord, elem);
            int64_t *result_ptr = reinterpret_cast<int64_t *>(coord[current_axis-1].data_ptr);
            *result_ptr = static_cast<int>(value);
        }
# else
        elem.dtp = ndt::make_type<int32_t>();
        arr = allocate_nd_arr(shape, coord, elem);
        int32_t *result_ptr = reinterpret_cast<int32_t *>(coord[current_axis-1].data_ptr);
        *result_ptr = static_cast<int>(value);
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
            *result_ptr = static_cast<int>(value);
        } else {
            elem.dtp = ndt::make_type<int64_t>();
            arr = allocate_nd_arr(shape, coord, elem);
            int64_t *result_ptr = reinterpret_cast<int64_t *>(coord[current_axis-1].data_ptr);
            *result_ptr = static_cast<int>(value);
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
        elem.dtp = ndt::make_type<double>();
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
                        if (coord[current_axis].coord >= coord[current_axis].reserved_size) {
                            // Increase the reserved capacity if needed
                            coord[current_axis].reserved_size *= 2;
                            char *data_ptr = (current_axis > 0) ? coord[current_axis-1].data_ptr
                                                                : arr.get_readwrite_originptr();
                            ndt::var_dim_element_resize(coord[current_axis].tp,
                                        coord[current_axis].metadata_ptr,
                                        data_ptr, coord[current_axis].reserved_size);
                        }
                    }
                    array_from_py_dynamic(item, shape, coord, elem,
                                arr, current_axis + 1);

                    item = PyIter_Next(iter);
                }

                if (PyErr_Occurred()) {
                    // Propagate any error
                    throw exception();
                }
            } else {
                // Because this iterator's sequence is zero-sized, we can't
                // start deducing the type yet. Thus, we make an array of
                // int32, but don't initialize elem yet.
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

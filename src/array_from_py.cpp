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
} // anonymous namespace

namespace {
    enum shape_signals_t {
        shape_signal_var = -1,
        shape_signal_ragged = -2,
        shape_signal_dict = -3,
        shape_signal_uninitialized = -4
    };
};

static void deduce_pylist_shape_and_ndt_type(PyObject *obj,
                vector<intptr_t>& shape, ndt::type& dt, size_t current_axis)
{
    if (PyList_Check(obj)) {
        Py_ssize_t size = PyList_GET_SIZE(obj);
        if (shape.size() == current_axis) {
            if (dt.get_type_id() == void_type_id) {
                shape.push_back(size);
            } else {
                throw runtime_error("dynd array doesn't support dimensions which are sometimes scalars and sometimes arrays");
            }
        } else {
            if (shape[current_axis] != size) {
                // A variable-sized dimension
                shape[current_axis] = shape_signal_var;
            }
        }
        
        for (Py_ssize_t i = 0; i < size; ++i) {
            deduce_pylist_shape_and_ndt_type(PyList_GET_ITEM(obj, i), shape, dt, current_axis + 1);
        }
    } else {
        if (shape.size() != current_axis) {
            throw runtime_error("dynd array doesn't support dimensions which are sometimes scalars and sometimes arrays");
        }

        ndt::type obj_dt;
#if PY_VERSION_HEX >= 0x03000000
        if (PyUnicode_Check(obj)) {
            obj_dt = ndt::make_string(string_encoding_utf_8);
        } else {
            obj_dt = pydynd::deduce_ndt_type_from_pyobject(obj);
        }
#else
        obj_dt = pydynd::deduce_ndt_type_from_pyobject(obj);
#endif

        if (dt != obj_dt) {
            dt = dynd::promote_types_arithmetic(obj_dt, dt);
        }
    }
}

/**
 * Returns the number of dimensions without raggedness, where
 * structs and array dimensions are non-scalar.
 */
static size_t get_nonragged_dim_count(const ndt::type& dt, size_t max_count=numeric_limits<size_t>::max())
{
    switch (dt.get_kind()) {    
        case uniform_dim_kind:
            if (max_count <= 1) {
                return max_count;
            } else {
                return min(max_count,
                        1 + get_nonragged_dim_count(
                            static_cast<const base_uniform_dim_type *>(
                                dt.extended())->get_element_type(), max_count - 1));
            }
        case struct_kind:
            if (max_count <= 1) {
                return max_count;
            } else {
                const base_struct_type *bsd = static_cast<const base_struct_type *>(dt.extended());
                size_t field_count = bsd->get_field_count();
                const ndt::type *field_types = bsd->get_field_types();
                for (size_t i = 0; i != field_count; ++i) {
                    size_t candidate = 1 + get_nonragged_dim_count(field_types[i], max_count - 1);
                    if (candidate < max_count) {
                        max_count = candidate;
                        if (max_count <= 1) {
                            return max_count;
                        }
                    }
                }
                return max_count;
            }
        default:
            return 0;
    }
}

static void deduce_pyseq_shape_with_dtype(PyObject *obj, const ndt::type& udt,
                std::vector<intptr_t>& shape, bool initial_pass, size_t current_axis)
{
    bool is_sequence = (PySequence_Check(obj) != 0 && !PyUnicode_Check(obj));
#if PY_VERSION_HEX < 0x03000000
    is_sequence = is_sequence && !PyString_Check(obj);
#endif
    Py_ssize_t size = 0;
    if (is_sequence) {
        size = PySequence_Size(obj);
        if (size == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            is_sequence = false;
        }
    }

    if (is_sequence) {
        if (shape.size() == current_axis) {
            if (initial_pass) {
                shape.push_back(size);
            } else if (udt.get_kind() == struct_kind) {
                // Signal that this is a dimension which is sometimes scalar, to allow for
                // raggedness in the struct type's fields
                shape.push_back(shape_signal_ragged);
            } else {
                throw runtime_error("dynd array doesn't support dimensions"
                                " which are sometimes scalars and sometimes arrays");
            }
        } else {
            if (shape[current_axis] != size && shape[current_axis] >= 0) {
                // A variable-sized dimension
                shape[current_axis] = shape_signal_var;
            }
        }

        for (Py_ssize_t i = 0; i < size; ++i) {
            pyobject_ownref item(PySequence_GetItem(obj, i));
            deduce_pyseq_shape_with_dtype(item.get(), udt,
                            shape, i == 0 && initial_pass, current_axis + 1);
        }
    } else {
        if (PyDict_Check(obj) && udt.get_kind() == struct_kind) {
            if (shape.size() == current_axis) {
                shape.push_back(shape_signal_dict);
            } else if (shape[current_axis] != shape_signal_ragged) {
                shape[current_axis] = shape_signal_dict;
            }
        } else if (shape.size() != current_axis) {
            if (udt.get_kind() == struct_kind) {
                shape[current_axis] = shape_signal_ragged;
            } else {
                throw runtime_error("dynd array doesn't support dimensions"
                                " which are sometimes scalars and sometimes arrays");
            }
        }
    }
}

static void deduce_pyseq_shape(PyObject *obj, size_t undim, intptr_t *shape)
{
    bool is_sequence = (PySequence_Check(obj) != 0);
    Py_ssize_t size = 0;
    if (is_sequence) {
        size = PySequence_Size(obj);
        if (size == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            is_sequence = false;
        }
    }

    if (is_sequence) {
        if (shape[0] == shape_signal_uninitialized) {
            shape[0] = size;
        } else if (shape[0] != size) {
            // A variable-sized dimension
            shape[0] = shape_signal_var;
        }

        if (undim > 1) {
            for (Py_ssize_t i = 0; i < size; ++i) {
                pyobject_ownref item(PySequence_GetItem(obj, i));
                deduce_pyseq_shape(item.get(), undim - 1, shape + 1);
            }
        }
    } else {
        // If it's an iterator, error checking needs to be done later
        // during actual value assignment.
        PyObject *iter = PyObject_GetIter(obj);
        if (iter == NULL) {
            if (PyErr_ExceptionMatches(PyExc_TypeError)) {
                PyErr_Clear();
                throw runtime_error("not enough dimensions in"
                                " python object for the provided dynd type");
            } else {
                // Propagate the exception
                throw exception();
            }
        } else {
            Py_DECREF(iter);
        }
        // It must be a variable-sized dimension
        shape[0] = shape_signal_var;
    }
}


typedef void (*convert_one_pyscalar_function_t)(const ndt::type& dt,
                const char *metadata, char *out, PyObject *obj);

inline void convert_one_pyscalar_bool(const ndt::type& dt, const char *metadata, char *out, PyObject *obj)
{
    *out = (PyObject_IsTrue(obj) != 0);
}

inline void convert_one_pyscalar_int32(const ndt::type& dt, const char *metadata, char *out, PyObject *obj)
{
#if PY_VERSION_HEX >= 0x03000000
    *reinterpret_cast<int32_t *>(out) = static_cast<int32_t>(PyLong_AsLong(obj));
#else
    *reinterpret_cast<int32_t *>(out) = static_cast<int32_t>(PyInt_AsLong(obj));
#endif
}

inline void convert_one_pyscalar_int64(const ndt::type& dt, const char *metadata, char *out, PyObject *obj)
{
    *reinterpret_cast<int64_t *>(out) = PyLong_AsLongLong(obj);
}

inline void convert_one_pyscalar_double(const ndt::type& dt, const char *metadata, char *out, PyObject *obj)
{
    *reinterpret_cast<double *>(out) = PyFloat_AsDouble(obj);
}

inline void convert_one_pyscalar_cdouble(const ndt::type& dt, const char *metadata, char *out, PyObject *obj)
{
    *reinterpret_cast<complex<double> *>(out) = complex<double>(
                    PyComplex_RealAsDouble(obj), PyComplex_ImagAsDouble(obj));
}

struct bytes_string_ptrs {
    char *begin, *end;
};

inline void convert_one_pyscalar_bytes(const ndt::type& dt, const char *metadata, char *out, PyObject *obj)
{
    bytes_string_ptrs *out_asp = reinterpret_cast<bytes_string_ptrs *>(out);
    const string_type_metadata *md = reinterpret_cast<const string_type_metadata *>(metadata);
#if PY_VERSION_HEX >= 0x03000000
    if (PyBytes_Check(obj)) {
#else
    if (PyString_Check(obj)) {
#endif
        char *data = NULL;
        intptr_t len = 0;
#if PY_VERSION_HEX >= 0x03000000
        if (PyBytes_AsStringAndSize(obj, &data, &len) < 0) {
#else
        if (PyString_AsStringAndSize(obj, &data, &len) < 0) {
#endif
            throw runtime_error("Error getting byte string data");
        }

        memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(md->blockref);
        allocator->allocate(md->blockref, len, 1, &out_asp->begin, &out_asp->end);
        memcpy(out_asp->begin, data, len);
    } else {
        throw runtime_error("wrong kind of string provided (require byte string for dynd bytes type)");
    }
}

struct pyunicode_string_ptrs {
    char *begin, *end;
};

inline void convert_one_pyscalar_ustring(const ndt::type& dt, const char *metadata, char *out, PyObject *obj)
{
    pyunicode_string_ptrs *out_usp = reinterpret_cast<pyunicode_string_ptrs *>(out);
    const string_type_metadata *md = reinterpret_cast<const string_type_metadata *>(metadata);
    if (PyUnicode_Check(obj)) {
        // Get it as UTF8
        pyobject_ownref utf8(PyUnicode_AsUTF8String(obj));
        char *s = NULL;
        Py_ssize_t len = 0;
        if (PyBytes_AsStringAndSize(utf8.get(), &s, &len) < 0) {
            throw exception();
        }
        memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(md->blockref);
        allocator->allocate(md->blockref, len,
                        1, &out_usp->begin, &out_usp->end);
        memcpy(out_usp->begin, s, len);
#if PY_VERSION_HEX < 0x03000000
    } else if (PyString_Check(obj)) {
        char *data = NULL;
        Py_ssize_t len = 0;
        if (PyString_AsStringAndSize(obj, &data, &len) < 0) {
            throw runtime_error("Error getting string data");
        }

        memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(md->blockref);
        allocator->allocate(md->blockref, len,
                        1, &out_usp->begin, &out_usp->end);
        for (Py_ssize_t i = 0; i < len; ++i) {
            // Only let valid ascii get through
            if ((unsigned char)data[i] >= 128) {
                throw string_decode_error(data+i, data+i+1, string_encoding_ascii);
            }
            out_usp->begin[i] = data[i];
        }
#endif
    } else {
        throw runtime_error("wrong kind of string provided");
    }
}

inline void convert_one_pyscalar_date(const ndt::type& dt, const char *metadata, char *out, PyObject *obj)
{
    if (!PyDate_Check(obj)) {
        throw runtime_error("input object is not a date as expected");
    }
    const date_type *dd = static_cast<const date_type *>(dt.extended());
    dd->set_ymd(metadata, out, assign_error_fractional, PyDateTime_GET_YEAR(obj),
                    PyDateTime_GET_MONTH(obj), PyDateTime_GET_DAY(obj));
}

inline void convert_one_pyscalar_datetime(const ndt::type& dt, const char *metadata, char *out, PyObject *obj)
{
    if (!PyDateTime_Check(obj)) {
        throw runtime_error("input object is not a datetime as expected");
    }
    if (((PyDateTime_DateTime *)obj)->hastzinfo &&
                    ((PyDateTime_DateTime *)obj)->tzinfo != NULL) {
        throw runtime_error("Converting datetimes with a timezone to dynd arrays is not yet supported");
    }
    const datetime_type *dd = static_cast<const datetime_type *>(dt.extended());
    dd->set_cal(metadata, out, assign_error_fractional, PyDateTime_GET_YEAR(obj),
                    PyDateTime_GET_MONTH(obj), PyDateTime_GET_DAY(obj),
                    PyDateTime_DATE_GET_HOUR(obj), PyDateTime_DATE_GET_MINUTE(obj),
                    PyDateTime_DATE_GET_SECOND(obj), PyDateTime_DATE_GET_MICROSECOND(obj) * 1000);
}

inline void convert_one_pyscalar_ndt_type(const ndt::type& DYND_UNUSED(dt),
                const char *DYND_UNUSED(metadata), char *out, PyObject *obj)
{
    ndt::type dt = make_ndt_type_from_pyobject(obj);
    dt.swap(reinterpret_cast<type_type_data *>(out)->dt);
}

template<convert_one_pyscalar_function_t ConvertOneFn>
static void fill_array_from_pylist(const ndt::type& dt, const char *metadata, char *data, PyObject *obj,
                const intptr_t *shape, size_t current_axis)
{
    if (shape[current_axis] == 0) {
        return;
    }

    Py_ssize_t size = PyList_GET_SIZE(obj);
    const char *element_metadata = metadata;
    ndt::type element_tp = dt.at_single(0, &element_metadata);
    if (shape[current_axis] >= 0) {
        // Fixed-sized dimension
        const strided_dim_type_metadata *md = reinterpret_cast<const strided_dim_type_metadata *>(metadata);
        intptr_t stride = md->stride;
        if (element_tp.is_scalar()) {
            for (Py_ssize_t i = 0; i < size; ++i) {
                PyObject *item = PyList_GET_ITEM(obj, i);
                ConvertOneFn(element_tp, element_metadata, data, item);
                data += stride;
            }
        } else {
            for (Py_ssize_t i = 0; i < size; ++i) {
                fill_array_from_pylist<ConvertOneFn>(element_tp, element_metadata, data,
                                PyList_GET_ITEM(obj, i), shape, current_axis + 1);
                data += stride;
            }
        }
    } else {
        // Variable-sized dimension
        const var_dim_type_metadata *md = reinterpret_cast<const var_dim_type_metadata *>(metadata);
        intptr_t stride = md->stride;
        var_dim_type_data *out = reinterpret_cast<var_dim_type_data *>(data);
        char *out_end = NULL;

        memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(md->blockref);
        allocator->allocate(md->blockref, size * stride,
                        element_tp.get_data_alignment(), &out->begin, &out_end);
        out->size = size;
        char *element_data = out->begin;
        if (element_tp.is_scalar()) {
            for (Py_ssize_t i = 0; i < size; ++i) {
                PyObject *item = PyList_GET_ITEM(obj, i);
                ConvertOneFn(element_tp, element_metadata, element_data, item);
                element_data += stride;
            }
        } else {
            for (Py_ssize_t i = 0; i < size; ++i) {
                fill_array_from_pylist<ConvertOneFn>(element_tp, element_metadata, element_data,
                                PyList_GET_ITEM(obj, i), shape, current_axis + 1);
                element_data += stride;
            }
        }
    }
}

static dynd::nd::array array_from_pylist(PyObject *obj)
{
    // TODO: Add ability to specify access flags (e.g. immutable)
    // Do a pass through all the data to deduce its type and shape
    vector<intptr_t> shape;
    ndt::type dt(void_type_id);
    Py_ssize_t size = PyList_GET_SIZE(obj);
    shape.push_back(size);
    for (Py_ssize_t i = 0; i < size; ++i) {
        deduce_pylist_shape_and_ndt_type(PyList_GET_ITEM(obj, i), shape, dt, 1);
    }
    // If no type was deduced, e.g. a size-zero list, default to double/float64
    if (dt.get_type_id() == void_type_id) {
        dt = ndt::make_type<double>();
    }

    // Create the array
    nd::array result = nd::make_strided_array(dt, (int)shape.size(), &shape[0],
                    nd::read_access_flag|nd::write_access_flag, NULL);

    // Populate the array with data
    switch (dt.get_type_id()) {
        case bool_type_id:
            fill_array_from_pylist<convert_one_pyscalar_bool>(result.get_type(), result.get_ndo_meta(),
                            result.get_readwrite_originptr(),
                            obj, &shape[0], 0);
            break;
        case int32_type_id:
            fill_array_from_pylist<convert_one_pyscalar_int32>(result.get_type(), result.get_ndo_meta(),
                            result.get_readwrite_originptr(),
                            obj, &shape[0], 0);
            break;
        case int64_type_id:
            fill_array_from_pylist<convert_one_pyscalar_int64>(result.get_type(), result.get_ndo_meta(),
                            result.get_readwrite_originptr(),
                            obj, &shape[0], 0);
            break;
        case float64_type_id:
            fill_array_from_pylist<convert_one_pyscalar_double>(result.get_type(), result.get_ndo_meta(),
                            result.get_readwrite_originptr(),
                            obj, &shape[0], 0);
            break;
        case complex_float64_type_id:
            fill_array_from_pylist<convert_one_pyscalar_cdouble>(result.get_type(), result.get_ndo_meta(),
                            result.get_readwrite_originptr(),
                            obj, &shape[0], 0);
            break;
        case bytes_type_id:
            fill_array_from_pylist<convert_one_pyscalar_bytes>(result.get_type(),
                            result.get_ndo_meta(),
                            result.get_readwrite_originptr(),
                            obj, &shape[0], 0);
            break;
        case string_type_id: {
            const base_string_type *ext = static_cast<const base_string_type *>(dt.extended());
            if (ext->get_encoding() == string_encoding_utf_8) {
                fill_array_from_pylist<convert_one_pyscalar_ustring>(result.get_type(),
                                result.get_ndo_meta(),
                                result.get_readwrite_originptr(),
                                obj, &shape[0], 0);
            } else {
                stringstream ss;
                ss << "Internal error: deduced type from Python list, " << dt << ", doesn't have a dynd array conversion";
                throw runtime_error(ss.str());
            }
            break;
        }
        case date_type_id: {
            fill_array_from_pylist<convert_one_pyscalar_date>(result.get_type(), result.get_ndo_meta(),
                            result.get_readwrite_originptr(),
                            obj, &shape[0], 0);
            break;
        }
        case datetime_type_id: {
            fill_array_from_pylist<convert_one_pyscalar_datetime>(result.get_type(), result.get_ndo_meta(),
                            result.get_readwrite_originptr(),
                            obj, &shape[0], 0);
            break;
        }
        case type_type_id: {
            fill_array_from_pylist<convert_one_pyscalar_ndt_type>(result.get_type(), result.get_ndo_meta(),
                            result.get_readwrite_originptr(),
                            obj, &shape[0], 0);
            break;
        }
        default: {
            stringstream ss;
            ss << "Deduced type from Python list, " << dt << ", doesn't have a dynd array conversion function yet";
            throw runtime_error(ss.str());
        }
    }
    result.get_type().extended()->metadata_finalize_buffers(result.get_ndo_meta());
    return result;
}

dynd::nd::array pydynd::array_from_py(PyObject *obj, uint32_t access_flags, bool always_copy)
{
    // If it's a Cython w_array
    if (WArray_Check(obj)) {
        const nd::array& result = ((WArray *)obj)->v;
        if (always_copy) {
            return result.eval_copy(access_flags);
        } else {
            if (access_flags != 0) {
                uint32_t raf = result.get_access_flags();
                if ((access_flags&nd::immutable_access_flag) && !(raf&nd::immutable_access_flag)) {
                    throw runtime_error("cannot view a non-immutable dynd array as immutable");
                }
                if ((access_flags&nd::write_access_flag) && !(raf&nd::write_access_flag)) {
                    throw runtime_error("cannot view a readonly dynd array as readwrite");
                }
            }
            return result;
        }
    }

#if DYND_NUMPY_INTEROP
    if (PyArray_Check(obj)) {
        return array_from_numpy_array((PyArrayObject *)obj, access_flags, always_copy);
    } else if (PyArray_IsScalar(obj, Generic)) {
        return array_from_numpy_scalar(obj, access_flags);
    }
#endif // DYND_NUMPY_INTEROP

    nd::array result;

    if (PyBool_Check(obj)) {
        result = nd::array(obj == Py_True);
#if PY_VERSION_HEX < 0x03000000
    } else if (PyInt_Check(obj)) {
        long value = PyInt_AS_LONG(obj);
# if SIZEOF_LONG > SIZEOF_INT
        // Use a 32-bit int if it fits. This conversion strategy
        // is independent of sizeof(long), and is the same on 32-bit
        // and 64-bit platforms.
        if (value >= INT_MIN && value <= INT_MAX) {
            result = static_cast<int>(value);
        } else {
            result = value;
        }
# else
        result = value;
# endif
#endif // PY_VERSION_HEX < 0x03000000
    } else if (PyLong_Check(obj)) {
        PY_LONG_LONG value = PyLong_AsLongLong(obj);
        if (value == -1 && PyErr_Occurred()) {
            throw runtime_error("error converting int value");
        }

        // Use a 32-bit int if it fits. This conversion strategy
        // is independent of sizeof(long), and is the same on 32-bit
        // and 64-bit platforms.
        if (value >= INT_MIN && value <= INT_MAX) {
            result = static_cast<int>(value);
        } else {
            result = value;
        }
    } else if (PyFloat_Check(obj)) {
        result = PyFloat_AS_DOUBLE(obj);
    } else if (PyComplex_Check(obj)) {
        result = complex<double>(PyComplex_RealAsDouble(obj), PyComplex_ImagAsDouble(obj));
#if PY_VERSION_HEX < 0x03000000
    } else if (PyString_Check(obj)) {
        char *data = NULL;
        Py_ssize_t len = 0;
        if (PyString_AsStringAndSize(obj, &data, &len) < 0) {
            throw runtime_error("Error getting string data");
        }

        for (Py_ssize_t i = 0; i < len; ++i) {
            // Only let valid ascii get through
            if ((unsigned char)data[i] >= 128) {
                throw string_decode_error(data+i, data+i+1, string_encoding_ascii);
            }
        }

        result = nd::make_utf8_array(data, len);
        
#else
    } else if (PyBytes_Check(obj)) {
        // Cannot provide write access unless a copy is being made
        if ((access_flags&nd::write_access_flag) != 0) {
            if (always_copy) {
                // If a readwrite copy is requested, make a new bytes array and copy the data.
                // For readonly copies, no need to copy because the data is immutable.
                char *data = NULL;
                intptr_t len = 0;
                if (PyBytes_AsStringAndSize(obj, &data, &len) < 0) {
                    throw runtime_error("Error getting byte string data");
                }
                result = nd::make_bytes_array(data, len);
                result.get_ndo()->m_flags = access_flags;
                return result;
            } else {
                throw runtime_error("cannot create a writable view of a python bytes object");
            }
        }

        char *data = NULL;
        intptr_t len = 0;
        if (PyBytes_AsStringAndSize(obj, &data, &len) < 0) {
            throw runtime_error("Error getting byte string data");
        }
        ndt::type d = ndt::make_bytes(1);
        // Python bytes are immutable, so simply use the existing memory with an external memory 
        Py_INCREF(obj);
        memory_block_ptr bytesref = make_external_memory_block(reinterpret_cast<void *>(obj), &py_decref_function);
        char *data_ptr;
        result = nd::array(make_array_memory_block(d.extended()->get_metadata_size(),
                        d.get_data_size(), d.get_data_alignment(), &data_ptr));
        result.get_ndo()->m_data_pointer = data_ptr;
        result.get_ndo()->m_data_reference = NULL;
        result.get_ndo()->m_type = d.extended();
        base_type_incref(result.get_ndo()->m_type);
        // The scalar consists of pointers to the byte string data
        ((const char **)data_ptr)[0] = data;
        ((const char **)data_ptr)[1] = data + len;
        // The metadata
        string_type_metadata *md = reinterpret_cast<string_type_metadata *>(result.get_ndo_meta());
        md->blockref = bytesref.release();
        result.get_ndo()->m_flags = nd::immutable_access_flag|nd::read_access_flag;
        // Because this is a view into another object's memory, skip the later processing
        return result;
#endif
    } else if (PyUnicode_Check(obj)) {
        pyobject_ownref utf8(PyUnicode_AsUTF8String(obj));
        char *s = NULL;
        Py_ssize_t len = 0;
        if (PyBytes_AsStringAndSize(utf8.get(), &s, &len) < 0) {
            throw exception();
        }
        result = nd::make_utf8_array(s, len);
    } else if (PyDateTime_Check(obj)) {
        if (((PyDateTime_DateTime *)obj)->hastzinfo &&
                        ((PyDateTime_DateTime *)obj)->tzinfo != NULL) {
            throw runtime_error("Converting datetimes with a timezone to dynd arrays is not yet supported");
        }
        ndt::type d = ndt::make_datetime(datetime_unit_usecond, tz_abstract);
        const datetime_type *dd = static_cast<const datetime_type *>(d.extended());
        result = nd::empty(d);
        dd->set_cal(result.get_ndo_meta(), result.get_ndo()->m_data_pointer, assign_error_fractional,
                    PyDateTime_GET_YEAR(obj), PyDateTime_GET_MONTH(obj), PyDateTime_GET_DAY(obj),
                    PyDateTime_DATE_GET_HOUR(obj), PyDateTime_DATE_GET_MINUTE(obj),
                    PyDateTime_DATE_GET_SECOND(obj), PyDateTime_DATE_GET_MICROSECOND(obj) * 1000);
    } else if (PyDate_Check(obj)) {
        ndt::type d = ndt::make_date();
        const date_type *dd = static_cast<const date_type *>(d.extended());
        result = nd::empty(d);
        dd->set_ymd(result.get_ndo_meta(), result.get_ndo()->m_data_pointer, assign_error_fractional,
                    PyDateTime_GET_YEAR(obj), PyDateTime_GET_MONTH(obj), PyDateTime_GET_DAY(obj));
    } else if (WType_Check(obj)) {
        result = nd::array(((WType *)obj)->v);
    } else if (PyList_Check(obj)) {
        result = array_from_pylist(obj);
    } else if (PyType_Check(obj)) {
        result = nd::array(make_ndt_type_from_pyobject(obj));
#if DYND_NUMPY_INTEROP
    } else if (PyArray_DescrCheck(obj)) {
        result = nd::array(make_ndt_type_from_pyobject(obj));
#endif // DYND_NUMPY_INTEROP
    }

    if (result.get_ndo() == NULL) {
        // Special case if it's an iterator: convert to a datashape of 'var, float64'.
        // This is based on NumPy's default np.fromiter(it) behavior when no type
        // is specified
        PyObject *iter = PyObject_GetIter(obj);
        if (iter != NULL) {
            // TODO: Maybe directly call the assign from pyiter function in array_assign_from_py.cpp
            Py_DECREF(iter);
            nd::array result = nd::empty(ndt::make_var_dim(ndt::make_type<double>()));
            array_nodim_broadcast_assign_from_py(result.get_type(),
                            result.get_ndo_meta(), result.get_readwrite_originptr(), obj);
            return result;
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

    if (result.get_ndo() == NULL) {
        throw std::runtime_error("could not convert python object into a dynd array");
    }

    // If write access wasn't specified, we can flag it as
    // immutable, because it's a newly allocated object.
    // This also covers the default case (access_flags==0),
    // which we want to be immutable as well.
    if ((access_flags&nd::write_access_flag) == 0) {
        result.flag_as_immutable();
    }

    return result;
}

static bool ndt_type_requires_shape(const ndt::type& dt)
{
    if (dt.get_ndim() > 0) {
        switch (dt.get_type_id()) {
            case fixed_dim_type_id:
            case var_dim_type_id:
                return ndt_type_requires_shape(
                                static_cast<const base_uniform_dim_type *>(
                                    dt.extended())->get_element_type());
            default:
                return true;
        }
    } else {
        return false;
    }
}

dynd::nd::array pydynd::array_from_py(PyObject *obj, const ndt::type& dt, bool uniform, uint32_t access_flags)
{
    nd::array result;
    if (uniform) {
        if (dt.get_ndim() != 0) {
            stringstream ss;
            ss << "dynd array creation with requested automatic deduction of array shape, but provided dt ";
            ss << dt << " already has an array dimension type";
            throw runtime_error(ss.str());
        }
        if (PyUnicode_Check(obj)
#if PY_VERSION_HEX < 0x03000000
                        || PyString_Check(obj)
#endif
                        ) {
            // Special case strings, because they act as sequences too
            result = nd::empty(dt);
        } else if (PySequence_Check(obj)) {
            vector<intptr_t> shape;
            Py_ssize_t size = PySequence_Size(obj);
            if (size == -1 && PyErr_Occurred()) {
                PyErr_Clear();
                result = nd::empty(dt);
            } else {
                shape.push_back(size);
                for (Py_ssize_t i = 0; i < size; ++i) {
                    pyobject_ownref item(PySequence_GetItem(obj, i));
                    deduce_pyseq_shape_with_dtype(item.get(), dt, shape, true, 1);
                }
                // If it's a struct, fix up the ndim to let the struct absorb
                // some of the sequences
                if (dt.get_kind() == struct_kind) {
                    size_t ndim, ndim_end = shape.size();
                    for (ndim = 0; ndim != ndim_end; ++ndim) {
                        if (shape[ndim] == shape_signal_ragged) {
                            // Match up the number of dimensions which aren't
                            // ragged in udt with the number of dimensions
                            // which are nonragged in the input data
                            size_t dt_nonragged = get_nonragged_dim_count(dt);
                            if (dt_nonragged <= ndim) {
                                ndim -= dt_nonragged;
                            } else {
                                ndim = 0;
                            }
                            break;
                        } else if (shape[ndim] == shape_signal_dict) {
                            break;
                        }
                    }
                    if (ndim == ndim_end) {
                        size_t dt_nonragged = get_nonragged_dim_count(dt);
                        if (dt_nonragged <= ndim) {
                            ndim -= dt_nonragged;
                        } else {
                            ndim = 0;
                        }
                    }
                    shape.resize(ndim);
                }

                result = nd::make_strided_array(dt, shape.size(), &shape[0]);
            }
        } else {
            // If the object is an iterator and the type doesn't already have
            // a array dimension, prepend a var dim as a special case
            PyObject *iter = PyObject_GetIter(obj);
            if (iter != NULL) {
                result = nd::empty(ndt::make_var_dim(dt));
            } else {
                if (PyErr_Occurred()) {
                    if (PyErr_ExceptionMatches(PyExc_TypeError)) {
                        // TypeError signals it doesn't support iteration
                        PyErr_Clear();
                    } else {
                        // Propagate errors
                        throw exception();
                    }
                }
                // If it wasn't a sequence or iterator, just use the type directly
                result = nd::empty(dt);
            }
        }
    } else if (dt.get_ndim() > 0 && ndt_type_requires_shape(dt)) {
        size_t undim = dt.get_ndim();
        dimvector shape(undim);
        for (size_t i = 0; i != undim; ++i) {
            shape[i] = shape_signal_uninitialized;
        }
        deduce_pyseq_shape(obj, undim, shape.get());
        result = nd::array(make_array_memory_block(dt, undim, shape.get()));
    } else {
        result = nd::empty(dt);
    }

    array_nodim_broadcast_assign_from_py(result.get_type(),
                    result.get_ndo_meta(), result.get_readwrite_originptr(), obj);


    // If write access wasn't requested, flag it as immutable
    if ((access_flags&nd::write_access_flag) == 0) {
        result.flag_as_immutable();
    }

    return result;
}

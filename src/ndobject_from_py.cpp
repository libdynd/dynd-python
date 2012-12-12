//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>
#include <datetime.h>

#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/dtypes/strided_array_dtype.hpp>
#include <dynd/dtypes/date_dtype.hpp>
#include <dynd/memblock/external_memory_block.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/dtype_promotion.hpp>

#include "ndobject_from_py.hpp"
#include "ndobject_functions.hpp"
#include "dtype_functions.hpp"
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

static void deduce_pylist_shape_and_dtype(PyObject *obj, vector<intptr_t>& shape, dtype& dt, int current_axis)
{
    if (PyList_Check(obj)) {
        Py_ssize_t size = PyList_GET_SIZE(obj);
        if (shape.size() == current_axis) {
            if (dt.get_type_id() == void_type_id) {
                shape.push_back(size);
            } else {
                throw runtime_error("dnd:ndobject doesn't support dimensions which are sometimes scalars and sometimes arrays");
            }
        } else {
            if (shape[current_axis] != size) {
                throw runtime_error("dynd::ndobject doesn't support arrays with varying dimension sizes yet");
            }
        }
        
        for (Py_ssize_t i = 0; i < size; ++i) {
            deduce_pylist_shape_and_dtype(PyList_GET_ITEM(obj, i), shape, dt, current_axis + 1);
        }
    } else {
        if (shape.size() != current_axis) {
            throw runtime_error("dnd:ndobject doesn't yet support dimensions which are sometimes scalars and sometimes arrays");
        }

        dtype obj_dt = pydynd::deduce_dtype_from_object(obj);
        if (dt != obj_dt) {
            dt = dynd::promote_dtypes_arithmetic(obj_dt, dt);
        }
    }
}

inline void convert_one_pyscalar(dynd_bool *out, PyObject *obj)
{
    *out = (PyObject_IsTrue(obj) != 0);
}

inline void convert_one_pyscalar(int32_t *out, PyObject *obj)
{
    *out = static_cast<int32_t>(PyInt_AsLong(obj));
}

inline void convert_one_pyscalar(int64_t *out, PyObject *obj)
{
    *out = PyLong_AsLongLong(obj);
}

inline void convert_one_pyscalar(double *out, PyObject *obj)
{
    *out = PyFloat_AsDouble(obj);
}

inline void convert_one_pyscalar(complex<double> *out, PyObject *obj)
{
    *out = complex<double>(PyComplex_RealAsDouble(obj), PyComplex_ImagAsDouble(obj));
}

struct ascii_string_ptrs {
    char *begin, *end;
};

inline void convert_one_string(const char *metadata, ascii_string_ptrs *out, PyObject *obj)
{
    const string_dtype_metadata *md = reinterpret_cast<const string_dtype_metadata *>(metadata);
    if (PyString_Check(obj)) {
        char *data = NULL;
        intptr_t len = 0;
        if (PyString_AsStringAndSize(obj, &data, &len) < 0) {
            throw runtime_error("Error getting string data");
        }

        memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(md->blockref);
        allocator->allocate(md->blockref, len, 1, &out->begin, &out->end);
        memcpy(out->begin, data, len);
    } else {
        throw runtime_error("wrong kind of string provided");
    }
}

inline void convert_one_date_pyscalar(const dtype& dt, char *out, PyObject *obj)
{
    if (!PyDate_Check(obj)) {
        throw runtime_error("input object is not a data as expected");
    }
    const date_dtype *dd = static_cast<const date_dtype *>(dt.extended());
    dd->set_ymd(NULL, out, assign_error_fractional, PyDateTime_GET_YEAR(obj),
                    PyDateTime_GET_MONTH(obj), PyDateTime_GET_DAY(obj));
}

struct pyunicode_string_ptrs {
#if Py_UNICODE_SIZE == 2
        uint16_t *begin, *end;
#else
        uint32_t *begin, *end;
#endif
};

inline void convert_one_string(const char *metadata, pyunicode_string_ptrs *out, PyObject *obj)
{
    const string_dtype_metadata *md = reinterpret_cast<const string_dtype_metadata *>(metadata);
    if (PyString_Check(obj)) {
        char *data = NULL;
        Py_ssize_t len = 0;
        if (PyString_AsStringAndSize(obj, &data, &len) < 0) {
            throw runtime_error("Error getting string data");
        }

        memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(md->blockref);
        allocator->allocate(md->blockref, len * Py_UNICODE_SIZE,
                        Py_UNICODE_SIZE, (char **)&out->begin, (char **)&out->end);
        for (Py_ssize_t i = 0; i < len; ++i) {
            out->begin[i] = data[i];
        }
    } else if (PyUnicode_Check(obj)) {
        const char *data = reinterpret_cast<const char *>(PyUnicode_AsUnicode(obj));
        Py_ssize_t len = PyUnicode_GetSize(obj);
        if (data == NULL || len == -1) {
            throw runtime_error("Error getting unicode string data");
        }
        memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(md->blockref);
        allocator->allocate(md->blockref, len * Py_UNICODE_SIZE,
                        Py_UNICODE_SIZE, (char **)&out->begin, (char **)&out->end);
        memcpy(out->begin, data, len * Py_UNICODE_SIZE);
    } else {
        throw runtime_error("wrong kind of string provided");
    }
}

template<typename T>
static T *fill_ndobject_from_pylist(T *data, PyObject *obj, const vector<intptr_t>& shape, int current_axis)
{
    if (current_axis == shape.size() - 1) {
        Py_ssize_t size = PyList_GET_SIZE(obj);
        for (Py_ssize_t i = 0; i < size; ++i) {
            PyObject *item = PyList_GET_ITEM(obj, i);
            convert_one_pyscalar(data, item);
            ++data;
        }
    } else {
        Py_ssize_t size = PyList_GET_SIZE(obj);
        for (Py_ssize_t i = 0; i < size; ++i) {
            data = fill_ndobject_from_pylist(data, PyList_GET_ITEM(obj, i), shape, current_axis + 1);
        }
    }
    return data;
}

template<typename T>
static T *fill_string_ndobject_from_pylist(const char *metadata, T *data, PyObject *obj, const vector<intptr_t>& shape,
                    int current_axis)
{
    metadata += sizeof(strided_array_dtype_metadata);

    if (current_axis == shape.size() - 1) {
        Py_ssize_t size = PyList_GET_SIZE(obj);
        for (Py_ssize_t i = 0; i < size; ++i) {
            PyObject *item = PyList_GET_ITEM(obj, i);
            convert_one_string(metadata, data, item);
            ++data;
        }
    } else {
        Py_ssize_t size = PyList_GET_SIZE(obj);
        for (Py_ssize_t i = 0; i < size; ++i) {
            data = fill_string_ndobject_from_pylist(metadata, data, PyList_GET_ITEM(obj, i), shape, current_axis + 1);
        }
    }
    return data;
}

char *fill_date_ndobject_from_pylist(const dtype& dt, char *data, PyObject *obj, const vector<intptr_t>& shape, int current_axis)
{
    if (current_axis == shape.size() - 1) {
        size_t element_size = dt.get_element_size();
        Py_ssize_t size = PyList_GET_SIZE(obj);
        for (Py_ssize_t i = 0; i < size; ++i) {
            PyObject *item = PyList_GET_ITEM(obj, i);
            convert_one_date_pyscalar(dt, data, item);
            data += element_size;
        }
    } else {
        Py_ssize_t size = PyList_GET_SIZE(obj);
        for (Py_ssize_t i = 0; i < size; ++i) {
            data = fill_date_ndobject_from_pylist(dt, data, PyList_GET_ITEM(obj, i), shape, current_axis + 1);
        }
    }
    return data;
}


static dynd::ndobject ndobject_from_pylist(PyObject *obj)
{
    // TODO: Add ability to specify access flags (e.g. immutable)
    // Do a pass through all the data to deduce its dtype and shape
    vector<intptr_t> shape;
    dtype dt;
    Py_ssize_t size = PyList_GET_SIZE(obj);
    shape.push_back(size);
    for (Py_ssize_t i = 0; i < size; ++i) {
        deduce_pylist_shape_and_dtype(PyList_GET_ITEM(obj, i), shape, dt, 1);
    }

    // Create the array
    ndobject result = make_strided_ndobject(dt, (int)shape.size(), &shape[0],
                    read_access_flag|write_access_flag, NULL);

    // Populate the array with data
    switch (dt.get_type_id()) {
        case bool_type_id:
            fill_ndobject_from_pylist(reinterpret_cast<dynd_bool *>(result.get_readwrite_originptr()),
                            obj, shape, 0);
            break;
        case int32_type_id:
            fill_ndobject_from_pylist(reinterpret_cast<int32_t *>(result.get_readwrite_originptr()),
                            obj, shape, 0);
            break;
        case int64_type_id:
            fill_ndobject_from_pylist(reinterpret_cast<int64_t *>(result.get_readwrite_originptr()),
                            obj, shape, 0);
            break;
        case float64_type_id:
            fill_ndobject_from_pylist(reinterpret_cast<double *>(result.get_readwrite_originptr()),
                            obj, shape, 0);
            break;
        case complex_float64_type_id:
            fill_ndobject_from_pylist(reinterpret_cast<complex<double> *>(result.get_readwrite_originptr()),
                            obj, shape, 0);
            break;
        case string_type_id: {
            const extended_string_dtype *ext = static_cast<const extended_string_dtype *>(dt.extended());
            switch (ext->get_encoding()) {
                case string_encoding_ascii:
                    fill_string_ndobject_from_pylist(result.get_ndo_meta(),
                                    reinterpret_cast<ascii_string_ptrs *>(result.get_readwrite_originptr()),
                                    obj, shape, 0);
                    result.get_dtype().extended()->metadata_finalize_buffers(result.get_ndo_meta());
                    break;
#if Py_UNICODE_SIZE == 2
                case string_encoding_ucs_2:
#else
                case string_encoding_utf_32:
#endif
                        {
                    fill_string_ndobject_from_pylist(result.get_ndo_meta(),
                                    reinterpret_cast<pyunicode_string_ptrs *>(result.get_readwrite_originptr()),
                                    obj, shape, 0);
                    result.get_dtype().extended()->metadata_finalize_buffers(result.get_ndo_meta());
                    break;
                }
                default:
                    stringstream ss;
                    ss << "Deduced type from Python list, " << dt << ", doesn't have a dynd::ndobject conversion function yet";
                    throw runtime_error(ss.str());
            }
            break;
        }
        case date_type_id: {
            fill_date_ndobject_from_pylist(dt, result.get_readwrite_originptr(), obj, shape, 0);
            break;
        }
        default: {
            stringstream ss;
            ss << "Deduced type from Python list, " << dt << ", doesn't have a dynd::ndobject conversion function yet";
            throw runtime_error(ss.str());
        }
    }

    return result;
}

dynd::ndobject pydynd::ndobject_from_py(PyObject *obj)
{
    // If it's a Cython w_ndobject
    if (WNDObject_Check(obj)) {
        return ((WNDObject *)obj)->v;
    }

#if DYND_NUMPY_INTEROP
    if (PyArray_Check(obj)) {
        return ndobject_from_numpy_array((PyArrayObject *)obj);
    } else if (PyArray_IsScalar(obj, Generic)) {
        return ndobject_from_numpy_scalar(obj);
    }
#endif // DYND_NUMPY_INTEROP

    if (PyBool_Check(obj)) {
        return ndobject(obj == Py_True);
#if PY_VERSION_HEX < 0x03000000
    } else if (PyInt_Check(obj)) {
        long value = PyInt_AS_LONG(obj);
# if SIZEOF_LONG > SIZEOF_INT
        // Use a 32-bit int if it fits. This conversion strategy
        // is independent of sizeof(long), and is the same on 32-bit
        // and 64-bit platforms.
        if (value >= INT_MIN && value <= INT_MAX) {
            return static_cast<int>(value);
        } else {
            return value;
        }
# else
        return value;
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
            return static_cast<int>(value);
        } else {
            return value;
        }
    } else if (PyFloat_Check(obj)) {
        return PyFloat_AS_DOUBLE(obj);
    } else if (PyComplex_Check(obj)) {
        return complex<double>(PyComplex_RealAsDouble(obj), PyComplex_ImagAsDouble(obj));
    } else if (PyString_Check(obj)) { // TODO: On Python 3, PyBytes should become a dnd bytes array
        char *data = NULL;
        intptr_t len = 0;
        if (PyString_AsStringAndSize(obj, &data, &len) < 0) {
            throw runtime_error("Error getting string data");
        }
        dtype d = make_string_dtype(string_encoding_ascii);
        const char *refs[2] = {data, data + len};
        // Python strings are immutable, so simply use the existing memory with an external memory 
        Py_INCREF(obj);
        memory_block_ptr stringref = make_external_memory_block(reinterpret_cast<void *>(obj), &py_decref_function);
        char *data_ptr;
        ndobject result(make_ndobject_memory_block(d.extended()->get_metadata_size(),
                        d.get_element_size(), d.get_alignment(), &data_ptr));
        result.get_ndo()->m_data_pointer = data_ptr;
        result.get_ndo()->m_data_reference = NULL;
        result.get_ndo()->m_dtype = d.extended();
        extended_dtype_incref(result.get_ndo()->m_dtype);
        // The scalar consists of pointers to the string data
        ((const char **)data_ptr)[0] = data;
        ((const char **)data_ptr)[1] = data + len;
        // The metadata
        string_dtype_metadata *md = reinterpret_cast<string_dtype_metadata *>(result.get_ndo_meta());
        md->blockref = stringref.release();
        result.get_ndo()->m_flags = immutable_access_flag|read_access_flag;
        return result;
    } else if (PyUnicode_Check(obj)) {
#if Py_UNICODE_SIZE == 2
        dtype d = make_string_dtype(string_encoding_ucs_2);
#else
        dtype d = make_string_dtype(string_encoding_utf_32);
#endif
        const char *data = reinterpret_cast<const char *>(PyUnicode_AsUnicode(obj));
        // Python strings are immutable, so simply use the existing memory with an external memory block
        Py_INCREF(obj);
        memory_block_ptr stringdata = make_external_memory_block(reinterpret_cast<void *>(obj), &py_decref_function);
        char *data_ptr;
        ndobject result(make_ndobject_memory_block(d.extended()->get_metadata_size(),
                    d.get_element_size(), d.get_alignment(), &data_ptr));
        result.get_ndo()->m_data_pointer = data_ptr;
        result.get_ndo()->m_data_reference = NULL;
        result.get_ndo()->m_dtype = d.extended();
        extended_dtype_incref(result.get_ndo()->m_dtype);
        // The scalar consists of pointers to the string data
        ((const char **)data_ptr)[0] = data;
        ((const char **)data_ptr)[1] = data + Py_UNICODE_SIZE * PyUnicode_GetSize(obj);
        // The metadata
        string_dtype_metadata *md = reinterpret_cast<string_dtype_metadata *>(result.get_ndo_meta());
        md->blockref = stringdata.release();
        result.get_ndo()->m_flags = immutable_access_flag|read_access_flag;
        return result;
    } else if (PyDate_Check(obj)) {
        dtype d = make_date_dtype();
        const date_dtype *dd = static_cast<const date_dtype *>(d.extended());
        ndobject result(d);
        dd->set_ymd(result.get_ndo_meta(), result.get_ndo()->m_data_pointer, assign_error_fractional,
                    PyDateTime_GET_YEAR(obj), PyDateTime_GET_MONTH(obj), PyDateTime_GET_DAY(obj));
        return result;
    } else if (PyList_Check(obj)) {
        return ndobject_from_pylist(obj);
    } else {
        throw std::runtime_error("could not convert python object into a dynd::ndobject");
    }
}

//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>
#include <datetime.h>

#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/dtypes/strided_dim_dtype.hpp>
#include <dynd/dtypes/fixed_dim_dtype.hpp>
#include <dynd/dtypes/var_dim_dtype.hpp>
#include <dynd/dtypes/base_struct_dtype.hpp>
#include <dynd/dtypes/date_dtype.hpp>
#include <dynd/dtypes/dtype_dtype.hpp>
#include <dynd/memblock/external_memory_block.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/dtype_promotion.hpp>

#include "ndobject_from_py.hpp"
#include "ndobject_assign_from_py.hpp"
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

namespace {
    enum shape_signals_t {
        shape_signal_var = -1,
        shape_signal_ragged = -2,
        shape_signal_dict = -3,
        shape_signal_uninitialized = -4
    };
};

static void deduce_pylist_shape_and_dtype(PyObject *obj,
                vector<intptr_t>& shape, dtype& dt, size_t current_axis)
{
    if (PyList_Check(obj)) {
        Py_ssize_t size = PyList_GET_SIZE(obj);
        if (shape.size() == current_axis) {
            if (dt.get_type_id() == void_type_id) {
                shape.push_back(size);
            } else {
                throw runtime_error("dynd ndobject doesn't support dimensions which are sometimes scalars and sometimes arrays");
            }
        } else {
            if (shape[current_axis] != size) {
                // A variable-sized dimension
                shape[current_axis] = shape_signal_var;
            }
        }
        
        for (Py_ssize_t i = 0; i < size; ++i) {
            deduce_pylist_shape_and_dtype(PyList_GET_ITEM(obj, i), shape, dt, current_axis + 1);
        }
    } else {
        if (shape.size() != current_axis) {
            throw runtime_error("dynd ndobject doesn't support dimensions which are sometimes scalars and sometimes arrays");
        }

        dtype obj_dt;
#if PY_VERSION_HEX >= 0x03000000
        if (PyUnicode_Check(obj)) {
            obj_dt = make_string_dtype(string_encoding_utf_8);
        } else {
            obj_dt = pydynd::deduce_dtype_from_pyobject(obj);
        }
#else
        obj_dt = pydynd::deduce_dtype_from_pyobject(obj);
#endif

        if (dt != obj_dt) {
            dt = dynd::promote_dtypes_arithmetic(obj_dt, dt);
        }
    }
}

/**
 * Returns the number of dimensions without raggedness, where
 * structs and uniform dimensions are non-scalar.
 */
static size_t get_nonragged_dim_count(const dtype& dt, size_t max_count=numeric_limits<size_t>::max())
{
    switch (dt.get_kind()) {    
        case uniform_dim_kind:
            if (max_count <= 1) {
                return max_count;
            } else {
                return min(max_count,
                        1 + get_nonragged_dim_count(
                            static_cast<const base_uniform_dim_dtype *>(
                                dt.extended())->get_element_dtype(), max_count - 1));
            }
        case struct_kind:
            if (max_count <= 1) {
                return max_count;
            } else {
                const base_struct_dtype *bsd = static_cast<const base_struct_dtype *>(dt.extended());
                size_t field_count = bsd->get_field_count();
                const dtype *field_types = bsd->get_field_types();
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

static void deduce_pyseq_shape_with_udtype(PyObject *obj, const dtype& udt,
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
                // raggedness in the struct dtype's fields
                shape.push_back(shape_signal_ragged);
            } else {
                throw runtime_error("dynd ndobject doesn't support dimensions"
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
            deduce_pyseq_shape_with_udtype(item.get(), udt,
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
                throw runtime_error("dynd ndobject doesn't support dimensions"
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
        throw runtime_error("not enough dimensions in"
                        " python object for the provided dynd type");
    }
}


typedef void (*convert_one_pyscalar_function_t)(const dtype& dt,
                const char *metadata, char *out, PyObject *obj);

inline void convert_one_pyscalar_bool(const dtype& dt, const char *metadata, char *out, PyObject *obj)
{
    *out = (PyObject_IsTrue(obj) != 0);
}

inline void convert_one_pyscalar_int32(const dtype& dt, const char *metadata, char *out, PyObject *obj)
{
#if PY_VERSION_HEX >= 0x03000000
    *reinterpret_cast<int32_t *>(out) = static_cast<int32_t>(PyLong_AsLong(obj));
#else
    *reinterpret_cast<int32_t *>(out) = static_cast<int32_t>(PyInt_AsLong(obj));
#endif
}

inline void convert_one_pyscalar_int64(const dtype& dt, const char *metadata, char *out, PyObject *obj)
{
    *reinterpret_cast<int64_t *>(out) = PyLong_AsLongLong(obj);
}

inline void convert_one_pyscalar_double(const dtype& dt, const char *metadata, char *out, PyObject *obj)
{
    *reinterpret_cast<double *>(out) = PyFloat_AsDouble(obj);
}

inline void convert_one_pyscalar_cdouble(const dtype& dt, const char *metadata, char *out, PyObject *obj)
{
    *reinterpret_cast<complex<double> *>(out) = complex<double>(
                    PyComplex_RealAsDouble(obj), PyComplex_ImagAsDouble(obj));
}

#if PY_VERSION_HEX < 0x03000000
struct ascii_string_ptrs {
    char *begin, *end;
};

inline void convert_one_pyscalar_astring(const dtype& dt, const char *metadata, char *out, PyObject *obj)
{
    ascii_string_ptrs *out_asp = reinterpret_cast<ascii_string_ptrs *>(out);
    const string_dtype_metadata *md = reinterpret_cast<const string_dtype_metadata *>(metadata);
    if (PyString_Check(obj)) {
        char *data = NULL;
        intptr_t len = 0;
        if (PyString_AsStringAndSize(obj, &data, &len) < 0) {
            throw runtime_error("Error getting string data");
        }

        memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(md->blockref);
        allocator->allocate(md->blockref, len, 1, &out_asp->begin, &out_asp->end);
        memcpy(out_asp->begin, data, len);
    } else {
        throw runtime_error("wrong kind of string provided");
    }
}
#endif

#if PY_VERSION_HEX >= 0x03030000
struct pyunicode_string_ptrs {
    char *begin, *end;
};
#else
struct pyunicode_string_ptrs {
#  if Py_UNICODE_SIZE == 2
    uint16_t *begin, *end;
#  else
    uint32_t *begin, *end;
#  endif
};
#endif

inline void convert_one_pyscalar_ustring(const dtype& dt, const char *metadata, char *out, PyObject *obj)
{
    pyunicode_string_ptrs *out_usp = reinterpret_cast<pyunicode_string_ptrs *>(out);
    const string_dtype_metadata *md = reinterpret_cast<const string_dtype_metadata *>(metadata);
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
                        1, (char **)&out_usp->begin, (char **)&out_usp->end);
        memcpy(out_usp->begin, s, len);
#if PY_VERSION_HEX < 0x03000000
    } else if (PyString_Check(obj)) {
        char *data = NULL;
        Py_ssize_t len = 0;
        if (PyString_AsStringAndSize(obj, &data, &len) < 0) {
            throw runtime_error("Error getting string data");
        }

        memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(md->blockref);
        allocator->allocate(md->blockref, len * Py_UNICODE_SIZE,
                        Py_UNICODE_SIZE, (char **)&out_usp->begin, (char **)&out_usp->end);
        for (Py_ssize_t i = 0; i < len; ++i) {
            out_usp->begin[i] = data[i];
        }
#endif
    } else {
        throw runtime_error("wrong kind of string provided");
    }
}

inline void convert_one_pyscalar_date(const dtype& dt, const char *metadata, char *out, PyObject *obj)
{
    if (!PyDate_Check(obj)) {
        throw runtime_error("input object is not a data as expected");
    }
    const date_dtype *dd = static_cast<const date_dtype *>(dt.extended());
    dd->set_ymd(metadata, out, assign_error_fractional, PyDateTime_GET_YEAR(obj),
                    PyDateTime_GET_MONTH(obj), PyDateTime_GET_DAY(obj));
}

inline void convert_one_pyscalar_dtype(const dtype& DYND_UNUSED(dt),
                const char *DYND_UNUSED(metadata), char *out, PyObject *obj)
{
    dtype dt = make_dtype_from_pyobject(obj);
    dt.swap(reinterpret_cast<dtype_dtype_data *>(out)->dt);
}

template<convert_one_pyscalar_function_t ConvertOneFn>
static void fill_ndobject_from_pylist(const dtype& dt, const char *metadata, char *data, PyObject *obj,
                const intptr_t *shape, size_t current_axis)
{
    Py_ssize_t size = PyList_GET_SIZE(obj);
    const char *element_metadata = metadata;
    dtype element_dtype = dt.at_single(0, &element_metadata);
    if (shape[current_axis] >= 0) {
        // Fixed-sized dimension
        const strided_dim_dtype_metadata *md = reinterpret_cast<const strided_dim_dtype_metadata *>(metadata);
        intptr_t stride = md->stride;
        if (element_dtype.is_scalar()) {
            for (Py_ssize_t i = 0; i < size; ++i) {
                PyObject *item = PyList_GET_ITEM(obj, i);
                ConvertOneFn(element_dtype, element_metadata, data, item);
                data += stride;
            }
        } else {
            for (Py_ssize_t i = 0; i < size; ++i) {
                fill_ndobject_from_pylist<ConvertOneFn>(element_dtype, element_metadata, data,
                                PyList_GET_ITEM(obj, i), shape, current_axis + 1);
                data += stride;
            }
        }
    } else {
        // Variable-sized dimension
        const var_dim_dtype_metadata *md = reinterpret_cast<const var_dim_dtype_metadata *>(metadata);
        intptr_t stride = md->stride;
        var_dim_dtype_data *out = reinterpret_cast<var_dim_dtype_data *>(data);
        char *out_end = NULL;

        memory_block_pod_allocator_api *allocator = get_memory_block_pod_allocator_api(md->blockref);
        allocator->allocate(md->blockref, size * stride,
                        element_dtype.get_alignment(), &out->begin, &out_end);
        out->size = size;
        char *element_data = out->begin;
        if (element_dtype.is_scalar()) {
            for (Py_ssize_t i = 0; i < size; ++i) {
                PyObject *item = PyList_GET_ITEM(obj, i);
                ConvertOneFn(element_dtype, element_metadata, element_data, item);
                element_data += stride;
            }
        } else {
            for (Py_ssize_t i = 0; i < size; ++i) {
                fill_ndobject_from_pylist<ConvertOneFn>(element_dtype, element_metadata, element_data,
                                PyList_GET_ITEM(obj, i), shape, current_axis + 1);
                element_data += stride;
            }
        }
    }
}

static dynd::ndobject ndobject_from_pylist(PyObject *obj)
{
    // TODO: Add ability to specify access flags (e.g. immutable)
    // Do a pass through all the data to deduce its dtype and shape
    vector<intptr_t> shape;
    dtype dt(void_type_id);
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
            fill_ndobject_from_pylist<convert_one_pyscalar_bool>(result.get_dtype(), result.get_ndo_meta(),
                            result.get_readwrite_originptr(),
                            obj, &shape[0], 0);
            break;
        case int32_type_id:
            fill_ndobject_from_pylist<convert_one_pyscalar_int32>(result.get_dtype(), result.get_ndo_meta(),
                            result.get_readwrite_originptr(),
                            obj, &shape[0], 0);
            break;
        case int64_type_id:
            fill_ndobject_from_pylist<convert_one_pyscalar_int64>(result.get_dtype(), result.get_ndo_meta(),
                            result.get_readwrite_originptr(),
                            obj, &shape[0], 0);
            break;
        case float64_type_id:
            fill_ndobject_from_pylist<convert_one_pyscalar_double>(result.get_dtype(), result.get_ndo_meta(),
                            result.get_readwrite_originptr(),
                            obj, &shape[0], 0);
            break;
        case complex_float64_type_id:
            fill_ndobject_from_pylist<convert_one_pyscalar_cdouble>(result.get_dtype(), result.get_ndo_meta(),
                            result.get_readwrite_originptr(),
                            obj, &shape[0], 0);
            break;
        case string_type_id: {
            const base_string_dtype *ext = static_cast<const base_string_dtype *>(dt.extended());
            switch (ext->get_encoding()) {
#if PY_VERSION_HEX < 0x03000000
                case string_encoding_ascii:
                    fill_ndobject_from_pylist<convert_one_pyscalar_astring>(result.get_dtype(), result.get_ndo_meta(),
                                    result.get_readwrite_originptr(),
                                    obj, &shape[0], 0);
                    break;
#endif
#if PY_VERSION_HEX >= 0x03030000
                case string_encoding_utf_8:
#elif Py_UNICODE_SIZE == 2
                case string_encoding_ucs_2:
#else
                case string_encoding_utf_32:
#endif
                        {
                    fill_ndobject_from_pylist<convert_one_pyscalar_ustring>(result.get_dtype(),
                                    result.get_ndo_meta(),
                                    result.get_readwrite_originptr(),
                                    obj, &shape[0], 0);
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
            fill_ndobject_from_pylist<convert_one_pyscalar_date>(result.get_dtype(), result.get_ndo_meta(),
                            result.get_readwrite_originptr(),
                            obj, &shape[0], 0);
            break;
        }
        case dtype_type_id: {
            fill_ndobject_from_pylist<convert_one_pyscalar_dtype>(result.get_dtype(), result.get_ndo_meta(),
                            result.get_readwrite_originptr(),
                            obj, &shape[0], 0);
            break;
        }
        default: {
            stringstream ss;
            ss << "Deduced type from Python list, " << dt << ", doesn't have a dynd::ndobject conversion function yet";
            throw runtime_error(ss.str());
        }
    }
    result.get_dtype().extended()->metadata_finalize_buffers(result.get_ndo_meta());
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
    // TODO: On Python 3, PyBytes should become a dynd bytes array
#if PY_VERSION_HEX < 0x03000000
    } else if (PyString_Check(obj)) {
        char *data = NULL;
        intptr_t len = 0;
        if (PyString_AsStringAndSize(obj, &data, &len) < 0) {
            throw runtime_error("Error getting string data");
        }
        dtype d = make_string_dtype(string_encoding_ascii);
        // Python strings are immutable, so simply use the existing memory with an external memory 
        Py_INCREF(obj);
        memory_block_ptr stringref = make_external_memory_block(reinterpret_cast<void *>(obj), &py_decref_function);
        char *data_ptr;
        ndobject result(make_ndobject_memory_block(d.extended()->get_metadata_size(),
                        d.get_data_size(), d.get_alignment(), &data_ptr));
        result.get_ndo()->m_data_pointer = data_ptr;
        result.get_ndo()->m_data_reference = NULL;
        result.get_ndo()->m_dtype = d.extended();
        base_dtype_incref(result.get_ndo()->m_dtype);
        // The scalar consists of pointers to the string data
        ((const char **)data_ptr)[0] = data;
        ((const char **)data_ptr)[1] = data + len;
        // The metadata
        string_dtype_metadata *md = reinterpret_cast<string_dtype_metadata *>(result.get_ndo_meta());
        md->blockref = stringref.release();
        result.get_ndo()->m_flags = immutable_access_flag|read_access_flag;
        return result;
#endif
    } else if (PyUnicode_Check(obj)) {
        pyobject_ownref utf8(PyUnicode_AsUTF8String(obj));
        char *s = NULL;
        Py_ssize_t len = 0;
        if (PyBytes_AsStringAndSize(utf8.get(), &s, &len) < 0) {
            throw exception();
        }
        return make_utf8_ndobject(s, len);
    } else if (PyDate_Check(obj)) {
        dtype d = make_date_dtype();
        const date_dtype *dd = static_cast<const date_dtype *>(d.extended());
        ndobject result = empty(d);
        dd->set_ymd(result.get_ndo_meta(), result.get_ndo()->m_data_pointer, assign_error_fractional,
                    PyDateTime_GET_YEAR(obj), PyDateTime_GET_MONTH(obj), PyDateTime_GET_DAY(obj));
        return result;
    } else if (WDType_Check(obj)) {
        return ndobject(((WDType *)obj)->v);
    } else if (PyList_Check(obj)) {
        return ndobject_from_pylist(obj);
    } else if (PyType_Check(obj)) {
        return ndobject(make_dtype_from_pyobject(obj));
#if DYND_NUMPY_INTEROP
    } else if (PyArray_DescrCheck(obj)) {
        return ndobject(make_dtype_from_pyobject(obj));
#endif // DYND_NUMPY_INTEROP
    }

    throw std::runtime_error("could not convert python object into a dynd::ndobject");
}

static bool dtype_requires_shape(const dtype& dt)
{
    if (dt.get_undim() > 0) {
        switch (dt.get_type_id()) {
            case fixed_dim_type_id:
            case var_dim_type_id:
                return dtype_requires_shape(
                                static_cast<const base_uniform_dim_dtype *>(
                                    dt.extended())->get_element_dtype());
            default:
                return true;
        }
    } else {
        return false;
    }
}

dynd::ndobject pydynd::ndobject_from_py(PyObject *obj, const dtype& dt, bool uniform)
{
    ndobject result;
    if (uniform) {
        if (dt.get_undim() != 0) {
            stringstream ss;
            ss << "ndobject creation with requested automatic deduction of ndobject shape, but provided dt ";
            ss << dt << " already has a uniform dim";
            throw runtime_error(ss.str());
        }
        if (PySequence_Check(obj)
#if PY_VERSION_HEX < 0x03000000
                        && !PyString_Check(obj)
#endif
                        && !PyUnicode_Check(obj)) {
            vector<intptr_t> shape;
            Py_ssize_t size = PySequence_Size(obj);
            if (size == -1 && PyErr_Occurred()) {
                PyErr_Clear();
                result = empty(dt);
            } else {
                shape.push_back(size);
                for (Py_ssize_t i = 0; i < size; ++i) {
                    pyobject_ownref item(PySequence_GetItem(obj, i));
                    deduce_pyseq_shape_with_udtype(item.get(), dt, shape, true, 1);
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

                result = make_strided_ndobject(dt, shape.size(), &shape[0]);
            }
        } else {
            result = empty(dt);
        }
    } else if (dt.get_undim() > 0 && dtype_requires_shape(dt)) {
        size_t undim = dt.get_undim();
        dimvector shape(undim);
        for (size_t i = 0; i != undim; ++i) {
            shape[i] = shape_signal_uninitialized;
        }
        deduce_pyseq_shape(obj, undim, shape.get());
        result = ndobject(make_ndobject_memory_block(dt, undim, shape.get()));
    } else {
        result = empty(dt);
    }

    ndobject_nodim_broadcast_assign_from_py(result.get_dtype(),
                    result.get_ndo_meta(), result.get_readwrite_originptr(), obj);
    return result;
}

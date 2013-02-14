//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "numpy_interop.hpp"

#if DYND_NUMPY_INTEROP

#include <dynd/dtypes/byteswap_dtype.hpp>
#include <dynd/dtypes/view_dtype.hpp>
#include <dynd/dtypes/dtype_alignment.hpp>
#include <dynd/dtypes/fixedstring_dtype.hpp>
#include <dynd/dtypes/strided_dim_dtype.hpp>
#include <dynd/dtypes/struct_dtype.hpp>
#include <dynd/dtypes/fixedstruct_dtype.hpp>
#include <dynd/dtypes/fixed_dim_dtype.hpp>
#include <dynd/memblock/external_memory_block.hpp>

#include "dtype_functions.hpp"
#include "ndobject_functions.hpp"
#include "utility_functions.hpp"

#include <numpy/arrayscalars.h>

using namespace std;
using namespace dynd;
using namespace pydynd;

dtype make_struct_dtype_from_numpy_struct(PyArray_Descr *d, size_t data_alignment)
{
    vector<dtype> field_types;
    vector<string> field_names;
    vector<size_t> field_offsets;

    if (!PyDataType_HASFIELDS(d)) {
        throw runtime_error("Tried to make a tuple dtype from a Numpy descr without fields");
    }

    PyObject *names = d->names;
    Py_ssize_t names_size = PyTuple_GET_SIZE(names);

    // The alignment must divide into the total element size,
    // shrink it until it does.
    while (!offset_is_aligned((size_t)d->elsize, data_alignment)) {
        data_alignment >>= 1;
    }

    for (Py_ssize_t i = 0; i < names_size; ++i) {
        PyObject *key = PyTuple_GET_ITEM(names, i);
        PyObject *tup = PyDict_GetItem(d->fields, key);
        PyArray_Descr *fld_dtype;
        PyObject *title;
        int offset = 0;
        if (!PyArg_ParseTuple(tup, "Oi|O", &fld_dtype, &offset, &title)) {
            throw runtime_error("Numpy struct dtype has corrupt data");
        }
        field_types.push_back(dtype_from_numpy_dtype(fld_dtype, data_alignment));
        // If the field isn't aligned enough, turn it into an unaligned type
        if (!offset_is_aligned(offset | data_alignment, field_types.back().get_alignment())) {
            field_types.back() = make_unaligned_dtype(field_types.back());
        }
        field_names.push_back(pystring_as_string(key));
        field_offsets.push_back(offset);
    }

    // Make a fixedstruct if possible, struct otherwise
    if (is_fixedstruct_compatible_offsets((int)field_types.size(), &field_types[0], &field_offsets[0], d->elsize)) {
        return make_fixedstruct_dtype(field_types, field_names);
    } else {
        return make_struct_dtype(field_types, field_names);
    }
}

dtype pydynd::dtype_from_numpy_dtype(PyArray_Descr *d, size_t data_alignment)
{
    dtype dt;

    if (data_alignment == 0) {
        data_alignment = d->alignment;
    }

    if (d->subarray) {
        dt = dtype_from_numpy_dtype(d->subarray->base, data_alignment);
        if (dt.get_data_size() == 0) {
            // If the element size isn't fixed, use the strided array
            int ndim = 1;
            if (PyTuple_Check(d->subarray->shape)) {
                ndim = (int)PyTuple_GET_SIZE(d->subarray->shape);
            }
            return make_strided_dim_dtype(dt, ndim);
        } else {
            // Otherwise make a fixedstruct array
            return dnd_make_fixed_dim_dtype(d->subarray->shape, dt, Py_None);
        }
    }

    switch (d->type_num) {
    case NPY_BOOL:
        dt = make_dtype<dynd_bool>();
        break;
    case NPY_BYTE:
        dt = make_dtype<npy_byte>();
        break;
    case NPY_UBYTE:
        dt = make_dtype<npy_ubyte>();
        break;
    case NPY_SHORT:
        dt = make_dtype<npy_short>();
        break;
    case NPY_USHORT:
        dt = make_dtype<npy_ushort>();
        break;
    case NPY_INT:
        dt = make_dtype<npy_int>();
        break;
    case NPY_UINT:
        dt = make_dtype<npy_uint>();
        break;
    case NPY_LONG:
        dt = make_dtype<npy_long>();
        break;
    case NPY_ULONG:
        dt = make_dtype<npy_ulong>();
        break;
    case NPY_LONGLONG:
        dt = make_dtype<npy_longlong>();
        break;
    case NPY_ULONGLONG:
        dt = make_dtype<npy_ulonglong>();
        break;
    case NPY_FLOAT:
        dt = make_dtype<float>();
        break;
    case NPY_DOUBLE:
        dt = make_dtype<double>();
        break;
    case NPY_CFLOAT:
        dt = make_dtype<complex<float> >();
        break;
    case NPY_CDOUBLE:
        dt = make_dtype<complex<double> >();
        break;
    case NPY_STRING:
        dt = make_fixedstring_dtype(d->elsize, string_encoding_ascii);
        break;
    case NPY_UNICODE:
        dt = make_fixedstring_dtype(d->elsize / 4, string_encoding_utf_32);
        break;
    case NPY_VOID:
        dt = make_struct_dtype_from_numpy_struct(d, data_alignment);
        break;
    default: {
        stringstream ss;
        ss << "unsupported Numpy dtype with type id " << d->type_num;
        throw runtime_error(ss.str());
        }
    }

    if (!PyArray_ISNBO(d->byteorder)) {
        dt = make_byteswap_dtype(dt);
    }

    // If the data this dtype is for isn't aligned enough,
    // make an unaligned version.
    if (data_alignment < dt.get_alignment()) {
        dt = make_unaligned_dtype(dt);
    }

    return dt;
}

void pydynd::fill_metadata_from_numpy_dtype(const dtype& dt, PyArray_Descr *d, char *metadata)
{
    switch (dt.get_type_id()) {
        case struct_type_id: {
            // In DyND, the struct offsets are part of the metadata instead of the dtype.
            // That's why we have to populate them here.
            PyObject *d_names = d->names;
            const struct_dtype *sdt = static_cast<const struct_dtype *>(dt.extended());
            const dtype *fields = sdt->get_field_types();
            const size_t *metadata_offsets = sdt->get_metadata_offsets();
            size_t field_count = sdt->get_field_count();
            size_t *offsets = reinterpret_cast<size_t *>(metadata);
            for (size_t i = 0; i < field_count; ++i) {
                PyObject *tup = PyDict_GetItem(d->fields, PyTuple_GET_ITEM(d_names, i));
                PyArray_Descr *fld_dtype;
                PyObject *title;
                int offset = 0;
                if (!PyArg_ParseTuple(tup, "Oi|O", &fld_dtype, &offset, &title)) {
                    throw runtime_error("Numpy struct dtype has corrupt data");
                }
                // Set the field offset in the output metadata
                offsets[i] = offset;
                // Fill the metadata for the field, if necessary
                if (!fields[i].is_builtin()) {
                    fill_metadata_from_numpy_dtype(fields[i], fld_dtype, metadata + metadata_offsets[i]);
                }
            }
            break;
        }
        case strided_dim_type_id: {
            // The Numpy subarray becomes a series of strided_dim_dtypes, so we
            // need to copy the strides into the metadata.
            dtype el;
            PyArray_ArrayDescr *adescr = d->subarray;
            if (adescr == NULL) {
                stringstream ss;
                ss << "Internal error building dynd metadata: Numpy dtype has NULL subarray corresponding to strided_dim type";
                throw runtime_error(ss.str());
            }
            int ndim;
            if (PyTuple_Check(adescr->shape)) {
                ndim = (int)PyTuple_GET_SIZE(adescr->shape);
                strided_dim_dtype_metadata *md = reinterpret_cast<strided_dim_dtype_metadata *>(metadata);
                intptr_t stride = adescr->base->elsize;
                el = dt;
                for (int i = ndim-1; i >= 0; --i) {
                    md[i].size = pyobject_as_index(PyTuple_GET_ITEM(adescr->shape, i));
                    md[i].stride = stride;
                    stride *= md[i].size;
                    el = static_cast<const strided_dim_dtype *>(el.extended())->get_element_dtype();
                }
                metadata += ndim * sizeof(strided_dim_dtype_metadata);
            } else {
                ndim = 1;
                strided_dim_dtype_metadata *md = reinterpret_cast<strided_dim_dtype_metadata *>(metadata);
                metadata += sizeof(strided_dim_dtype_metadata);
                md->size = pyobject_as_index(adescr->shape);
                md->stride = adescr->base->elsize;
                el = static_cast<const strided_dim_dtype *>(dt.extended())->get_element_dtype();
            }
            // Fill the metadata for the array element, if necessary
            if (!el.is_builtin()) {
                fill_metadata_from_numpy_dtype(el, adescr->base, metadata);
            }
            break;
        }
        default:
            break;
    }
}


PyArray_Descr *pydynd::numpy_dtype_from_dtype(const dynd::dtype& dt)
{
    switch (dt.get_type_id()) {
        case bool_type_id:
            return PyArray_DescrFromType(NPY_BOOL);
        case int8_type_id:
            return PyArray_DescrFromType(NPY_INT8);
        case int16_type_id:
            return PyArray_DescrFromType(NPY_INT16);
        case int32_type_id:
            return PyArray_DescrFromType(NPY_INT32);
        case int64_type_id:
            return PyArray_DescrFromType(NPY_INT64);
        case uint8_type_id:
            return PyArray_DescrFromType(NPY_UINT8);
        case uint16_type_id:
            return PyArray_DescrFromType(NPY_UINT16);
        case uint32_type_id:
            return PyArray_DescrFromType(NPY_UINT32);
        case uint64_type_id:
            return PyArray_DescrFromType(NPY_UINT64);
        case float32_type_id:
            return PyArray_DescrFromType(NPY_FLOAT32);
        case float64_type_id:
            return PyArray_DescrFromType(NPY_FLOAT64);
        case complex_float32_type_id:
            return PyArray_DescrFromType(NPY_CFLOAT);
        case complex_float64_type_id:
            return PyArray_DescrFromType(NPY_CDOUBLE);
        case fixedstring_type_id: {
            const fixedstring_dtype *fdt = static_cast<const fixedstring_dtype *>(dt.extended());
            PyArray_Descr *result;
            switch (fdt->get_encoding()) {
                case string_encoding_ascii:
                    result = PyArray_DescrNewFromType(NPY_STRING);
                    result->elsize = (int)fdt->get_data_size();
                    return result;
                case string_encoding_utf_32:
                    result = PyArray_DescrNewFromType(NPY_UNICODE);
                    result->elsize = (int)fdt->get_data_size();
                    return result;
                default:
                    break;
            }
            break;
        }
        /*
        case tuple_type_id: {
            const tuple_dtype *tdt = static_cast<const tuple_dtype *>(dt.extended());
            const vector<dtype>& fields = tdt->get_fields();
            size_t num_fields = fields.size();
            const vector<size_t>& offsets = tdt->get_offsets();

            // TODO: Deal with the names better
            pyobject_ownref names_obj(PyList_New(num_fields));
            for (size_t i = 0; i < num_fields; ++i) {
                stringstream ss;
                ss << "f" << i;
                PyList_SET_ITEM((PyObject *)names_obj, i, PyString_FromString(ss.str().c_str()));
            }

            pyobject_ownref formats_obj(PyList_New(num_fields));
            for (size_t i = 0; i < num_fields; ++i) {
                PyList_SET_ITEM((PyObject *)formats_obj, i, (PyObject *)numpy_dtype_from_dtype(fields[i]));
            }

            pyobject_ownref offsets_obj(PyList_New(num_fields));
            for (size_t i = 0; i < num_fields; ++i) {
                PyList_SET_ITEM((PyObject *)offsets_obj, i, PyLong_FromSize_t(offsets[i]));
            }

            pyobject_ownref itemsize_obj(PyLong_FromSize_t(dt.get_data_size()));

            pyobject_ownref dict_obj(PyDict_New());
            PyDict_SetItemString(dict_obj, "names", names_obj);
            PyDict_SetItemString(dict_obj, "formats", formats_obj);
            PyDict_SetItemString(dict_obj, "offsets", offsets_obj);
            PyDict_SetItemString(dict_obj, "itemsize", itemsize_obj);

            PyArray_Descr *result = NULL;
            if (PyArray_DescrConverter(dict_obj, &result) != NPY_SUCCEED) {
                throw runtime_error("failed to convert tuple dtype into numpy dtype via dict");
            }
            return result;
        }
        */
        case fixedstruct_type_id: {
            const fixedstruct_dtype *tdt = static_cast<const fixedstruct_dtype *>(dt.extended());
            const dtype *field_types = tdt->get_field_types();
            const string *field_names = tdt->get_field_names();
            const vector<size_t>& offsets = tdt->get_data_offsets_vector();
            size_t field_count = tdt->get_field_count();

            pyobject_ownref names_obj(PyList_New(field_count));
            for (size_t i = 0; i < field_count; ++i) {
                PyList_SET_ITEM((PyObject *)names_obj, i, PyString_FromString(field_names[i].c_str()));
            }

            pyobject_ownref formats_obj(PyList_New(field_count));
            for (size_t i = 0; i < field_count; ++i) {
                PyList_SET_ITEM((PyObject *)formats_obj, i, (PyObject *)numpy_dtype_from_dtype(field_types[i]));
            }

            pyobject_ownref offsets_obj(PyList_New(field_count));
            for (size_t i = 0; i < field_count; ++i) {
                PyList_SET_ITEM((PyObject *)offsets_obj, i, PyLong_FromSize_t(offsets[i]));
            }

            pyobject_ownref itemsize_obj(PyLong_FromSize_t(dt.get_data_size()));

            pyobject_ownref dict_obj(PyDict_New());
            PyDict_SetItemString(dict_obj, "names", names_obj);
            PyDict_SetItemString(dict_obj, "formats", formats_obj);
            PyDict_SetItemString(dict_obj, "offsets", offsets_obj);
            PyDict_SetItemString(dict_obj, "itemsize", itemsize_obj);

            PyArray_Descr *result = NULL;
            if (PyArray_DescrConverter(dict_obj, &result) != NPY_SUCCEED) {
                stringstream ss;
                ss << "failed to convert dtype " << dt << " into numpy dtype via dict";
                throw runtime_error(ss.str());
            }
            return result;
        }
        case fixed_dim_type_id: {
            dtype child_dt = dt;
            vector<intptr_t> shape;
            do {
                const fixed_dim_dtype *tdt = static_cast<const fixed_dim_dtype *>(child_dt.extended());
                shape.push_back(tdt->get_fixed_dim_size());
                if (child_dt.get_data_size() != tdt->get_element_dtype().get_data_size() * shape.back()) {
                    stringstream ss;
                    ss << "Cannot convert dynd dtype " << dt << " into a numpy dtype because it is not C-order";
                    throw runtime_error(ss.str());
                }
                child_dt = tdt->get_element_dtype();
            } while (child_dt.get_type_id() == fixed_dim_type_id);
            pyobject_ownref dtype_obj((PyObject *)numpy_dtype_from_dtype(child_dt));
            pyobject_ownref shape_obj(intptr_array_as_tuple((int)shape.size(), &shape[0]));
            pyobject_ownref tuple_obj(PyTuple_New(2));
            PyTuple_SET_ITEM(tuple_obj.get(), 0, dtype_obj.release());
            PyTuple_SET_ITEM(tuple_obj.get(), 1, shape_obj.release());

            PyArray_Descr *result = NULL;
            if (PyArray_DescrConverter(tuple_obj, &result) != NPY_SUCCEED) {
                throw runtime_error("failed to convert dynd dtype into numpy subarray dtype");
            }
            return result;
        }
        case view_type_id: {
            // If there's a view which is for alignment purposes, throw it
            // away because Numpy works differently
            if (dt.operand_dtype().get_type_id() == fixedbytes_type_id) {
                return numpy_dtype_from_dtype(dt.value_dtype());
            }
            break;
        }
        case byteswap_type_id: {
            // If it's a simple byteswap from bytes, that can be converted
            if (dt.operand_dtype().get_type_id() == fixedbytes_type_id) {
                PyArray_Descr *unswapped = numpy_dtype_from_dtype(dt.value_dtype());
                PyArray_Descr *result = PyArray_DescrNewByteorder(unswapped, NPY_SWAP);
                Py_DECREF(unswapped);
                return result;
            }
        }
        default:
            break;
    }

    stringstream ss;
    ss << "cannot convert dynd dtype " << dt << " into a Numpy dtype";
    throw runtime_error(ss.str());
}

PyArray_Descr *pydynd::numpy_dtype_from_dtype(const dynd::dtype& dt, const char *metadata)
{
    switch (dt.get_type_id()) {
        case struct_type_id: {
            if (metadata == NULL) {
                stringstream ss;
                ss << "Can only convert dynd dtype " << dt << " into a numpy dtype with ndobject metadata";
                throw runtime_error(ss.str());
            }
            const struct_dtype *sdt = static_cast<const struct_dtype *>(dt.extended());
            const dtype *field_types = sdt->get_field_types();
            const string *field_names = sdt->get_field_names();
            const size_t *metadata_offsets = sdt->get_metadata_offsets();
            const size_t *offsets = sdt->get_data_offsets(metadata);
            size_t field_count = sdt->get_field_count();

            pyobject_ownref names_obj(PyList_New(field_count));
            for (size_t i = 0; i < field_count; ++i) {
                PyList_SET_ITEM((PyObject *)names_obj, i, PyString_FromString(field_names[i].c_str()));
            }

            pyobject_ownref formats_obj(PyList_New(field_count));
            for (size_t i = 0; i < field_count; ++i) {
                PyList_SET_ITEM((PyObject *)formats_obj, i,
                                (PyObject *)numpy_dtype_from_dtype(field_types[i], metadata + metadata_offsets[i]));
            }

            pyobject_ownref offsets_obj(PyList_New(field_count));
            for (size_t i = 0; i < field_count; ++i) {
                PyList_SET_ITEM((PyObject *)offsets_obj, i, PyLong_FromSize_t(offsets[i]));
            }

            pyobject_ownref itemsize_obj(PyLong_FromSize_t(dt.get_data_size()));

            pyobject_ownref dict_obj(PyDict_New());
            PyDict_SetItemString(dict_obj, "names", names_obj);
            PyDict_SetItemString(dict_obj, "formats", formats_obj);
            PyDict_SetItemString(dict_obj, "offsets", offsets_obj);
            PyDict_SetItemString(dict_obj, "itemsize", itemsize_obj);

            PyArray_Descr *result = NULL;
            if (PyArray_DescrConverter(dict_obj, &result) != NPY_SUCCEED) {
                throw runtime_error("failed to convert dtype into numpy struct dtype via dict");
            }
            return result;
        }
        default:
            return numpy_dtype_from_dtype(dt);
    }
}

int pydynd::dtype_from_numpy_scalar_typeobject(PyTypeObject* obj, dynd::dtype& out_d)
{
    if (obj == &PyBoolArrType_Type) {
        out_d = make_dtype<dynd_bool>();
    } else if (obj == &PyByteArrType_Type) {
        out_d = make_dtype<npy_byte>();
    } else if (obj == &PyUByteArrType_Type) {
        out_d = make_dtype<npy_ubyte>();
    } else if (obj == &PyShortArrType_Type) {
        out_d = make_dtype<npy_short>();
    } else if (obj == &PyUShortArrType_Type) {
        out_d = make_dtype<npy_ushort>();
    } else if (obj == &PyIntArrType_Type) {
        out_d = make_dtype<npy_int>();
    } else if (obj == &PyUIntArrType_Type) {
        out_d = make_dtype<npy_uint>();
    } else if (obj == &PyLongArrType_Type) {
        out_d = make_dtype<npy_long>();
    } else if (obj == &PyULongArrType_Type) {
        out_d = make_dtype<npy_ulong>();
    } else if (obj == &PyLongLongArrType_Type) {
        out_d = make_dtype<npy_longlong>();
    } else if (obj == &PyULongLongArrType_Type) {
        out_d = make_dtype<npy_ulonglong>();
    } else if (obj == &PyFloatArrType_Type) {
        out_d = make_dtype<npy_float>();
    } else if (obj == &PyDoubleArrType_Type) {
        out_d = make_dtype<npy_double>();
    } else if (obj == &PyCFloatArrType_Type) {
        out_d = make_dtype<complex<float> >();
    } else if (obj == &PyCDoubleArrType_Type) {
        out_d = make_dtype<complex<double> >();
    } else {
        return -1;
    }

    return 0;
}

dtype pydynd::dtype_of_numpy_scalar(PyObject* obj)
{
    if (PyArray_IsScalar(obj, Bool)) {
        return make_dtype<dynd_bool>();
    } else if (PyArray_IsScalar(obj, Byte)) {
        return make_dtype<npy_byte>();
    } else if (PyArray_IsScalar(obj, UByte)) {
        return make_dtype<npy_ubyte>();
    } else if (PyArray_IsScalar(obj, Short)) {
        return make_dtype<npy_short>();
    } else if (PyArray_IsScalar(obj, UShort)) {
        return make_dtype<npy_ushort>();
    } else if (PyArray_IsScalar(obj, Int)) {
        return make_dtype<npy_int>();
    } else if (PyArray_IsScalar(obj, UInt)) {
        return make_dtype<npy_uint>();
    } else if (PyArray_IsScalar(obj, Long)) {
        return make_dtype<npy_long>();
    } else if (PyArray_IsScalar(obj, ULong)) {
        return make_dtype<npy_ulong>();
    } else if (PyArray_IsScalar(obj, LongLong)) {
        return make_dtype<npy_longlong>();
    } else if (PyArray_IsScalar(obj, ULongLong)) {
        return make_dtype<npy_ulonglong>();
    } else if (PyArray_IsScalar(obj, Float)) {
        return make_dtype<float>();
    } else if (PyArray_IsScalar(obj, Double)) {
        return make_dtype<double>();
    } else if (PyArray_IsScalar(obj, CFloat)) {
        return make_dtype<complex<float> >();
    } else if (PyArray_IsScalar(obj, CDouble)) {
        return make_dtype<complex<double> >();
    }

    throw std::runtime_error("could not deduce a pydynd dtype from the numpy scalar object");
}

inline size_t get_alignment_of(uintptr_t align_bits)
{
    size_t alignment = 1;
    // Loop 4 times, maximum alignment of 16
    for (int i = 0; i < 4; ++i) {
        if ((align_bits & alignment) == 0) {
            alignment <<= 1;
        } else {
            return alignment;
        }
    }
    return alignment;
}

inline size_t get_alignment_of(PyArrayObject* obj)
{
    // Get the alignment of the data
    uintptr_t align_bits = reinterpret_cast<uintptr_t>(PyArray_DATA(obj));
    int ndim = PyArray_NDIM(obj);
    intptr_t *strides = PyArray_STRIDES(obj);
    for (int idim = 0; idim < ndim; ++idim) {
        align_bits |= (uintptr_t)strides[idim];
    }

    return get_alignment_of(align_bits);
}

ndobject pydynd::ndobject_from_numpy_array(PyArrayObject* obj)
{
    // Get the dtype of the array
    dtype d = pydynd::dtype_from_numpy_dtype(PyArray_DESCR(obj), get_alignment_of(obj));

    // Get a shared pointer that tracks buffer ownership
    PyObject *base = PyArray_BASE(obj);
    memory_block_ptr memblock;
    if (base == NULL || (PyArray_FLAGS(obj)&NPY_ARRAY_UPDATEIFCOPY) != 0) {
        Py_INCREF(obj);
        memblock = make_external_memory_block(obj, py_decref_function);
    } else {
        if (WNDObject_Check(base)) {
            // If the base of the numpy array is an ndobject, skip the Python reference
            memblock = ((WNDObject *)base)->v.get_data_memblock();
        } else {
            Py_INCREF(base);
            memblock = make_external_memory_block(base, py_decref_function);
        }
    }

    // Create the result ndobject
    char *metadata = NULL;
    ndobject result = make_strided_ndobject_from_data(d, PyArray_NDIM(obj),
                    PyArray_DIMS(obj), PyArray_STRIDES(obj),
                    read_access_flag | (PyArray_ISWRITEABLE(obj) ? write_access_flag : 0),
                    PyArray_BYTES(obj), DYND_MOVE(memblock), &metadata);
    if (d.get_type_id() == struct_type_id) {
        // If it's a struct, there's additional metadata that needs to be populated
        pydynd::fill_metadata_from_numpy_dtype(d, PyArray_DESCR(obj), metadata);
    }
    return result;
}

dynd::ndobject pydynd::ndobject_from_numpy_scalar(PyObject* obj)
{
    if (PyArray_IsScalar(obj, Bool)) {
        return ndobject((dynd_bool)(((PyBoolScalarObject *)obj)->obval != 0));
    } else if (PyArray_IsScalar(obj, Byte)) {
        return ndobject(((PyByteScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, UByte)) {
        return ndobject(((PyUByteScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, Short)) {
        return ndobject(((PyShortScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, UShort)) {
        return ndobject(((PyUShortScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, Int)) {
        return ndobject(((PyIntScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, UInt)) {
        return ndobject(((PyUIntScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, Long)) {
        return ndobject(((PyLongScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, ULong)) {
        return ndobject(((PyULongScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, LongLong)) {
        return ndobject(((PyLongLongScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, ULongLong)) {
        return ndobject(((PyULongLongScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, Float)) {
        return ndobject(((PyFloatScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, Double)) {
        return ndobject(((PyDoubleScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, CFloat)) {
        npy_cfloat& val = ((PyCFloatScalarObject *)obj)->obval;
        return ndobject(complex<float>(val.real, val.imag));
    } else if (PyArray_IsScalar(obj, CDouble)) {
        npy_cdouble& val = ((PyCDoubleScalarObject *)obj)->obval;
        return ndobject(complex<double>(val.real, val.imag));
    }

    throw std::runtime_error("could not create a dynd::ndobject from the numpy scalar object");
}

char pydynd::numpy_kindchar_of(const dynd::dtype& d)
{
    switch (d.get_kind()) {
    case bool_kind:
        return 'b';
    case int_kind:
        return 'i';
    case uint_kind:
        return 'u';
    case real_kind:
        return 'f';
    case complex_kind:
        return 'c';
    case string_kind:
        if (d.get_type_id() == fixedstring_type_id) {
            const base_string_dtype *esd = static_cast<const base_string_dtype *>(d.extended());
            switch (esd->get_encoding()) {
                case string_encoding_ascii:
                    return 'S';
                case string_encoding_utf_32:
                    return 'U';
                default:
                    break;
            }
        }
        break;
    default:
        break;
    }

    stringstream ss;
    ss << "dynd::dtype \"" << d << "\" does not have an equivalent numpy kind";
    throw runtime_error(ss.str());
}

#endif // DYND_NUMPY_INTEROP

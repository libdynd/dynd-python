//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "numpy_interop.hpp"

#if DYND_NUMPY_INTEROP

#include <dynd/types/byteswap_type.hpp>
#include <dynd/types/view_type.hpp>
#include <dynd/types/type_alignment.hpp>
#include <dynd/types/fixedstring_type.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/cstruct_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/memblock/external_memory_block.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/property_type.hpp>

#include "type_functions.hpp"
#include "array_functions.hpp"
#include "utility_functions.hpp"

#include <numpy/arrayscalars.h>

using namespace std;
using namespace dynd;
using namespace pydynd;

ndt::type make_struct_type_from_numpy_struct(PyArray_Descr *d, size_t data_alignment)
{
    vector<ndt::type> field_types;
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
        field_types.push_back(ndt_type_from_numpy_dtype(fld_dtype, data_alignment));
        // If the field isn't aligned enough, turn it into an unaligned type
        if (!offset_is_aligned(offset | data_alignment, field_types.back().get_data_alignment())) {
            field_types.back() = make_unaligned(field_types.back());
        }
        field_names.push_back(pystring_as_string(key));
        field_offsets.push_back(offset);
    }

    // Make a cstruct if possible, struct otherwise
    if (is_cstruct_compatible_offsets(field_types.size(),
                    &field_types[0], &field_offsets[0], d->elsize)) {
        return ndt::make_cstruct(field_types.size(), &field_types[0], &field_names[0]);
    } else {
        return ndt::make_struct(field_types, field_names);
    }
}

ndt::type pydynd::ndt_type_from_numpy_dtype(PyArray_Descr *d, size_t data_alignment)
{
    ndt::type dt;

    // We ignore d->alignment, because on some platforms dynd's alignmkent
    // is more strict than the platform/numpy's alignment.
    // E.g. On 32-bit linux, int64 is aligned to 8-bytes in dynd,
    // but 4-bytes on the platform.

    if (d->subarray) {
        dt = ndt_type_from_numpy_dtype(d->subarray->base, data_alignment);
        if (dt.get_data_size() == 0) {
            // If the element size isn't fixed, use the strided array
            int ndim = 1;
            if (PyTuple_Check(d->subarray->shape)) {
                ndim = (int)PyTuple_GET_SIZE(d->subarray->shape);
            }
            return ndt::make_strided_dim(dt, ndim);
        } else {
            // Otherwise make a cstruct array
            return dynd_make_fixed_dim_type(d->subarray->shape, dt, Py_None);
        }
    }

    switch (d->type_num) {
    case NPY_BOOL:
        dt = ndt::make_type<dynd_bool>();
        break;
    case NPY_BYTE:
        dt = ndt::make_type<npy_byte>();
        break;
    case NPY_UBYTE:
        dt = ndt::make_type<npy_ubyte>();
        break;
    case NPY_SHORT:
        dt = ndt::make_type<npy_short>();
        break;
    case NPY_USHORT:
        dt = ndt::make_type<npy_ushort>();
        break;
    case NPY_INT:
        dt = ndt::make_type<npy_int>();
        break;
    case NPY_UINT:
        dt = ndt::make_type<npy_uint>();
        break;
    case NPY_LONG:
        dt = ndt::make_type<npy_long>();
        break;
    case NPY_ULONG:
        dt = ndt::make_type<npy_ulong>();
        break;
    case NPY_LONGLONG:
        dt = ndt::make_type<npy_longlong>();
        break;
    case NPY_ULONGLONG:
        dt = ndt::make_type<npy_ulonglong>();
        break;
    case NPY_FLOAT:
        dt = ndt::make_type<float>();
        break;
    case NPY_DOUBLE:
        dt = ndt::make_type<double>();
        break;
    case NPY_CFLOAT:
        dt = ndt::make_type<complex<float> >();
        break;
    case NPY_CDOUBLE:
        dt = ndt::make_type<complex<double> >();
        break;
    case NPY_STRING:
        dt = ndt::make_fixedstring(d->elsize, string_encoding_ascii);
        break;
    case NPY_UNICODE:
        dt = ndt::make_fixedstring(d->elsize / 4, string_encoding_utf_32);
        break;
    case NPY_VOID:
        dt = make_struct_type_from_numpy_struct(d, data_alignment);
        break;
#if NPY_API_VERSION >= 6 // At least NumPy 1.6
    case NPY_DATETIME: {
        // Get the dtype info through the CPython API, slower
        // but lets NumPy's datetime API change without issue.
        pyobject_ownref mod(PyImport_ImportModule("numpy"));
        pyobject_ownref dd(PyObject_CallMethod(mod.get(),
                        const_cast<char *>("datetime_data"), const_cast<char *>("O"), d));
        pyobject_ownref unit(PyTuple_GetItem(dd.get(), 0));
        string s = pystring_as_string(unit.get());
        if (s == "D") {
            // If it's 'datetime64[D]', then use a dynd date dtype, with the needed adapter
            dt = ndt::make_reversed_property(ndt::make_date(),
                            ndt::make_type<int64_t>(), "days_after_1970_int64");
        }
        break;
    }
#endif // At least NumPy 1.6
    default:
        break;
    }

    if (dt.get_type_id() == uninitialized_type_id) {
        stringstream ss;
        ss << "unsupported Numpy dtype with type id " << d->type_num;
        throw runtime_error(ss.str());
    }

    if (!PyArray_ISNBO(d->byteorder)) {
        dt = ndt::make_byteswap(dt);
    }

    // If the data this dtype is for isn't aligned enough,
    // make an unaligned version.
    if (data_alignment != 0 && data_alignment < dt.get_data_alignment()) {
        dt = make_unaligned(dt);
    }

    return dt;
}

void pydynd::fill_metadata_from_numpy_dtype(const ndt::type& dt, PyArray_Descr *d, char *metadata)
{
    switch (dt.get_type_id()) {
        case struct_type_id: {
            // In DyND, the struct offsets are part of the metadata instead of the dtype.
            // That's why we have to populate them here.
            PyObject *d_names = d->names;
            const struct_type *sdt = static_cast<const struct_type *>(dt.extended());
            const ndt::type *fields = sdt->get_field_types();
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
            // The Numpy subarray becomes a series of strided_dim_types, so we
            // need to copy the strides into the metadata.
            ndt::type el;
            PyArray_ArrayDescr *adescr = d->subarray;
            if (adescr == NULL) {
                stringstream ss;
                ss << "Internal error building dynd metadata: Numpy dtype has NULL subarray corresponding to strided_dim type";
                throw runtime_error(ss.str());
            }
            if (PyTuple_Check(adescr->shape)) {
                int ndim = (int)PyTuple_GET_SIZE(adescr->shape);
                strided_dim_type_metadata *md = reinterpret_cast<strided_dim_type_metadata *>(metadata);
                intptr_t stride = adescr->base->elsize;
                el = dt;
                for (int i = ndim-1; i >= 0; --i) {
                    md[i].size = pyobject_as_index(PyTuple_GET_ITEM(adescr->shape, i));
                    md[i].stride = stride;
                    stride *= md[i].size;
                    el = static_cast<const strided_dim_type *>(el.extended())->get_element_type();
                }
                metadata += ndim * sizeof(strided_dim_type_metadata);
            } else {
                strided_dim_type_metadata *md = reinterpret_cast<strided_dim_type_metadata *>(metadata);
                metadata += sizeof(strided_dim_type_metadata);
                md->size = pyobject_as_index(adescr->shape);
                md->stride = adescr->base->elsize;
                el = static_cast<const strided_dim_type *>(dt.extended())->get_element_type();
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


PyArray_Descr *pydynd::numpy_dtype_from_ndt_type(const dynd::ndt::type& dt)
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
            const fixedstring_type *fdt = static_cast<const fixedstring_type *>(dt.extended());
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
            const tuple_type *tdt = static_cast<const tuple_type *>(dt.extended());
            const vector<ndt::type>& fields = tdt->get_fields();
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
                PyList_SET_ITEM((PyObject *)formats_obj, i, (PyObject *)numpy_dtype_from_ndt_type(fields[i]));
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
        case cstruct_type_id: {
            const cstruct_type *tdt = static_cast<const cstruct_type *>(dt.extended());
            const ndt::type *field_types = tdt->get_field_types();
            const string *field_names = tdt->get_field_names();
            const vector<size_t>& offsets = tdt->get_data_offsets_vector();
            size_t field_count = tdt->get_field_count();

            pyobject_ownref names_obj(PyList_New(field_count));
            for (size_t i = 0; i < field_count; ++i) {
#if PY_VERSION_HEX >= 0x03000000
                pyobject_ownref name_str(PyUnicode_FromStringAndSize(
                                field_names[i].data(), field_names[i].size()));
#else
                pyobject_ownref name_str(PyString_FromStringAndSize(
                                field_names[i].data(), field_names[i].size()));
#endif
                PyList_SET_ITEM((PyObject *)names_obj, i, name_str.release());
            }

            pyobject_ownref formats_obj(PyList_New(field_count));
            for (size_t i = 0; i < field_count; ++i) {
                PyList_SET_ITEM((PyObject *)formats_obj, i, (PyObject *)numpy_dtype_from_ndt_type(field_types[i]));
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
            ndt::type child_dt = dt;
            vector<intptr_t> shape;
            do {
                const fixed_dim_type *tdt = static_cast<const fixed_dim_type *>(child_dt.extended());
                shape.push_back(tdt->get_fixed_dim_size());
                if (child_dt.get_data_size() != tdt->get_element_type().get_data_size() * shape.back()) {
                    stringstream ss;
                    ss << "Cannot convert dynd type " << dt << " into a numpy dtype because it is not C-order";
                    throw runtime_error(ss.str());
                }
                child_dt = tdt->get_element_type();
            } while (child_dt.get_type_id() == fixed_dim_type_id);
            pyobject_ownref dtype_obj((PyObject *)numpy_dtype_from_ndt_type(child_dt));
            pyobject_ownref shape_obj(intptr_array_as_tuple((int)shape.size(), &shape[0]));
            pyobject_ownref tuple_obj(PyTuple_New(2));
            PyTuple_SET_ITEM(tuple_obj.get(), 0, dtype_obj.release());
            PyTuple_SET_ITEM(tuple_obj.get(), 1, shape_obj.release());

            PyArray_Descr *result = NULL;
            if (PyArray_DescrConverter(tuple_obj, &result) != NPY_SUCCEED) {
                throw runtime_error("failed to convert dynd type into numpy subarray dtype");
            }
            return result;
        }
        case view_type_id: {
            // If there's a view which is for alignment purposes, throw it
            // away because Numpy works differently
            if (dt.operand_type().get_type_id() == fixedbytes_type_id) {
                return numpy_dtype_from_ndt_type(dt.value_type());
            }
            break;
        }
        case byteswap_type_id: {
            // If it's a simple byteswap from bytes, that can be converted
            if (dt.operand_type().get_type_id() == fixedbytes_type_id) {
                PyArray_Descr *unswapped = numpy_dtype_from_ndt_type(dt.value_type());
                PyArray_Descr *result = PyArray_DescrNewByteorder(unswapped, NPY_SWAP);
                Py_DECREF(unswapped);
                return result;
            }
        }
        default:
            break;
    }

    stringstream ss;
    ss << "cannot convert dynd type " << dt << " into a Numpy dtype";
    throw runtime_error(ss.str());
}

PyArray_Descr *pydynd::numpy_dtype_from_ndt_type(const dynd::ndt::type& dt, const char *metadata)
{
    switch (dt.get_type_id()) {
        case struct_type_id: {
            if (metadata == NULL) {
                stringstream ss;
                ss << "Can only convert dynd type " << dt << " into a numpy dtype with array metadata";
                throw runtime_error(ss.str());
            }
            const struct_type *sdt = static_cast<const struct_type *>(dt.extended());
            const ndt::type *field_types = sdt->get_field_types();
            const string *field_names = sdt->get_field_names();
            const size_t *metadata_offsets = sdt->get_metadata_offsets();
            const size_t *offsets = sdt->get_data_offsets(metadata);
            size_t field_count = sdt->get_field_count();

            pyobject_ownref names_obj(PyList_New(field_count));
            for (size_t i = 0; i < field_count; ++i) {
#if PY_VERSION_HEX >= 0x03000000
                pyobject_ownref name_str(PyUnicode_FromStringAndSize(
                                field_names[i].data(), field_names[i].size()));
#else
                pyobject_ownref name_str(PyString_FromStringAndSize(
                                field_names[i].data(), field_names[i].size()));
#endif
                PyList_SET_ITEM((PyObject *)names_obj, i, name_str.release());
            }

            pyobject_ownref formats_obj(PyList_New(field_count));
            for (size_t i = 0; i < field_count; ++i) {
                PyList_SET_ITEM((PyObject *)formats_obj, i,
                                (PyObject *)numpy_dtype_from_ndt_type(field_types[i], metadata + metadata_offsets[i]));
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
            return numpy_dtype_from_ndt_type(dt);
    }
}

int pydynd::ndt_type_from_numpy_scalar_typeobject(PyTypeObject* obj, dynd::ndt::type& out_d)
{
    if (obj == &PyBoolArrType_Type) {
        out_d = ndt::make_type<dynd_bool>();
    } else if (obj == &PyByteArrType_Type) {
        out_d = ndt::make_type<npy_byte>();
    } else if (obj == &PyUByteArrType_Type) {
        out_d = ndt::make_type<npy_ubyte>();
    } else if (obj == &PyShortArrType_Type) {
        out_d = ndt::make_type<npy_short>();
    } else if (obj == &PyUShortArrType_Type) {
        out_d = ndt::make_type<npy_ushort>();
    } else if (obj == &PyIntArrType_Type) {
        out_d = ndt::make_type<npy_int>();
    } else if (obj == &PyUIntArrType_Type) {
        out_d = ndt::make_type<npy_uint>();
    } else if (obj == &PyLongArrType_Type) {
        out_d = ndt::make_type<npy_long>();
    } else if (obj == &PyULongArrType_Type) {
        out_d = ndt::make_type<npy_ulong>();
    } else if (obj == &PyLongLongArrType_Type) {
        out_d = ndt::make_type<npy_longlong>();
    } else if (obj == &PyULongLongArrType_Type) {
        out_d = ndt::make_type<npy_ulonglong>();
    } else if (obj == &PyFloatArrType_Type) {
        out_d = ndt::make_type<npy_float>();
    } else if (obj == &PyDoubleArrType_Type) {
        out_d = ndt::make_type<npy_double>();
    } else if (obj == &PyCFloatArrType_Type) {
        out_d = ndt::make_type<complex<float> >();
    } else if (obj == &PyCDoubleArrType_Type) {
        out_d = ndt::make_type<complex<double> >();
    } else {
        return -1;
    }

    return 0;
}

ndt::type pydynd::ndt_type_of_numpy_scalar(PyObject* obj)
{
    if (PyArray_IsScalar(obj, Bool)) {
        return ndt::make_type<dynd_bool>();
    } else if (PyArray_IsScalar(obj, Byte)) {
        return ndt::make_type<npy_byte>();
    } else if (PyArray_IsScalar(obj, UByte)) {
        return ndt::make_type<npy_ubyte>();
    } else if (PyArray_IsScalar(obj, Short)) {
        return ndt::make_type<npy_short>();
    } else if (PyArray_IsScalar(obj, UShort)) {
        return ndt::make_type<npy_ushort>();
    } else if (PyArray_IsScalar(obj, Int)) {
        return ndt::make_type<npy_int>();
    } else if (PyArray_IsScalar(obj, UInt)) {
        return ndt::make_type<npy_uint>();
    } else if (PyArray_IsScalar(obj, Long)) {
        return ndt::make_type<npy_long>();
    } else if (PyArray_IsScalar(obj, ULong)) {
        return ndt::make_type<npy_ulong>();
    } else if (PyArray_IsScalar(obj, LongLong)) {
        return ndt::make_type<npy_longlong>();
    } else if (PyArray_IsScalar(obj, ULongLong)) {
        return ndt::make_type<npy_ulonglong>();
    } else if (PyArray_IsScalar(obj, Float)) {
        return ndt::make_type<float>();
    } else if (PyArray_IsScalar(obj, Double)) {
        return ndt::make_type<double>();
    } else if (PyArray_IsScalar(obj, CFloat)) {
        return ndt::make_type<complex<float> >();
    } else if (PyArray_IsScalar(obj, CDouble)) {
        return ndt::make_type<complex<double> >();
    }

    throw std::runtime_error("could not deduce a pydynd type from the numpy scalar object");
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

nd::array pydynd::array_from_numpy_array(PyArrayObject* obj, uint32_t access_flags, bool always_copy)
{
    // If a copy isn't requested, make sure the access flags are ok
    if (!always_copy) {
        if ((access_flags&nd::write_access_flag) && !PyArray_ISWRITEABLE(obj)) {
            throw runtime_error("cannot view a readonly numpy array as readwrite");
        }
        if (access_flags&nd::immutable_access_flag) {
            throw runtime_error("cannot view a numpy array as immutable");
        }
    }

    // Get the dtype of the array
    ndt::type d = pydynd::ndt_type_from_numpy_dtype(PyArray_DESCR(obj), get_alignment_of(obj));

    // Get a shared pointer that tracks buffer ownership
    PyObject *base = PyArray_BASE(obj);
    memory_block_ptr memblock;
    if (base == NULL || (PyArray_FLAGS(obj)&NPY_ARRAY_UPDATEIFCOPY) != 0) {
        Py_INCREF(obj);
        memblock = make_external_memory_block(obj, py_decref_function);
    } else {
        if (WArray_CheckExact(base)) {
            // If the base of the numpy array is an nd::array, skip the Python reference
            memblock = ((WArray *)base)->v.get_data_memblock();
        } else {
            Py_INCREF(base);
            memblock = make_external_memory_block(base, py_decref_function);
        }
    }

    // Create the result nd::array
    char *metadata = NULL;
    nd::array result = nd::make_strided_array_from_data(d, PyArray_NDIM(obj),
                    PyArray_DIMS(obj), PyArray_STRIDES(obj),
                    nd::read_access_flag | (PyArray_ISWRITEABLE(obj) ? nd::write_access_flag : 0),
                    PyArray_BYTES(obj), DYND_MOVE(memblock), &metadata);
    if (d.get_type_id() == struct_type_id) {
        // If it's a struct, there's additional metadata that needs to be populated
        pydynd::fill_metadata_from_numpy_dtype(d, PyArray_DESCR(obj), metadata);
    }

    if (always_copy) {
        return result.eval_copy(access_flags);
    } else {
        if (access_flags != 0) {
            // Use the requested access flags
            result.get_ndo()->m_flags = access_flags;
        }
        return result;
    }
}

dynd::nd::array pydynd::array_from_numpy_scalar(PyObject* obj, uint32_t access_flags)
{
    nd::array result;
    if (PyArray_IsScalar(obj, Bool)) {
        result = nd::array((dynd_bool)(((PyBoolScalarObject *)obj)->obval != 0));
    } else if (PyArray_IsScalar(obj, Byte)) {
        result = nd::array(((PyByteScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, UByte)) {
        result = nd::array(((PyUByteScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, Short)) {
        result = nd::array(((PyShortScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, UShort)) {
        result = nd::array(((PyUShortScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, Int)) {
        result = nd::array(((PyIntScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, UInt)) {
        result = nd::array(((PyUIntScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, Long)) {
        result = nd::array(((PyLongScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, ULong)) {
        result = nd::array(((PyULongScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, LongLong)) {
        result = nd::array(((PyLongLongScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, ULongLong)) {
        result = nd::array(((PyULongLongScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, Float)) {
        result = nd::array(((PyFloatScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, Double)) {
        result = nd::array(((PyDoubleScalarObject *)obj)->obval);
    } else if (PyArray_IsScalar(obj, CFloat)) {
        npy_cfloat& val = ((PyCFloatScalarObject *)obj)->obval;
        result = nd::array(complex<float>(val.real, val.imag));
    } else if (PyArray_IsScalar(obj, CDouble)) {
        npy_cdouble& val = ((PyCDoubleScalarObject *)obj)->obval;
        result = nd::array(complex<double>(val.real, val.imag));
#if NPY_API_VERSION >= 6 // At least NumPy 1.6
    } else if (PyArray_IsScalar(obj, Datetime)) {
        const PyDatetimeScalarObject *scalar = (PyDatetimeScalarObject *)obj;
        int64_t val = scalar->obval;
        if (scalar->obmeta.base == NPY_FR_D) {
            result = nd::empty(ndt::make_date());
            if (val == NPY_DATETIME_NAT) {
                *reinterpret_cast<int32_t *>(result.get_readwrite_originptr()) =
                            DYND_DATE_NA;
            } else {
                *reinterpret_cast<int32_t *>(result.get_readwrite_originptr()) =
                            static_cast<int32_t>(val);
            }
        } else {
            throw runtime_error("Unsupported NumPy datetime unit");
        }
#endif
    } else {
        throw std::runtime_error("could not create a dynd array from the numpy scalar object");
    }

    result.get_ndo()->m_flags = access_flags ? access_flags : nd::default_access_flags;

    return result;
}

char pydynd::numpy_kindchar_of(const dynd::ndt::type& d)
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
            const base_string_type *esd = static_cast<const base_string_type *>(d.extended());
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
    ss << "dynd type \"" << d << "\" does not have an equivalent numpy kind";
    throw runtime_error(ss.str());
}

#endif // DYND_NUMPY_INTEROP

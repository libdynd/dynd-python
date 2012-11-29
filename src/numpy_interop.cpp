//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "numpy_interop.hpp"

#if DYND_NUMPY_INTEROP

#include <dynd/dtypes/byteswap_dtype.hpp>
#include <dynd/dtypes/view_dtype.hpp>
#include <dynd/dtypes/dtype_alignment.hpp>
#include <dynd/dtypes/fixedstring_dtype.hpp>
#include <dynd/dtypes/strided_array_dtype.hpp>
#include <dynd/dtypes/struct_dtype.hpp>
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
    vector<dtype> fields;
    vector<string> field_names;

    if (!PyDataType_HASFIELDS(d)) {
        throw runtime_error("Tried to make a tuple dtype from a Numpy descr without fields");
    }

    PyObject *names = d->names;
    Py_ssize_t names_size = PyTuple_GET_SIZE(names);
    size_t max_field_alignment = 1;

    // The alignment must divide into the total element size,
    // shrink it until it does.
    while ((((size_t)d->elsize)&(data_alignment-1)) != 0) {
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
        fields.push_back(dtype_from_numpy_dtype(fld_dtype));
        // If the field isn't aligned enough, turn it into an unaligned type
        if ((((offset | data_alignment) & (fields.back().get_alignment() - 1))) != 0) {
            fields.back() = make_unaligned_dtype(fields.back());
        }
        field_names.push_back(pystring_as_string(key));
    }

    return make_struct_dtype(fields, field_names);
}

dtype pydynd::dtype_from_numpy_dtype(PyArray_Descr *d, size_t data_alignment)
{
    dtype dt;

    if (data_alignment == 0) {
        data_alignment = d->alignment;
    }

    if (d->subarray) {
        int ndim = 1;
        if (PyTuple_Check(d->subarray->shape)) {
            ndim = (int)PyTuple_GET_SIZE(d->subarray->shape);
        }
        dt = dtype_from_numpy_dtype(d->subarray->base, data_alignment);
        return make_strided_array_dtype(dt, ndim);
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
        dt = make_fixedstring_dtype(string_encoding_ascii, d->elsize);
        break;
    case NPY_UNICODE:
        dt = make_fixedstring_dtype(string_encoding_utf_32, d->elsize / 4);
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
            const vector<dtype>& fields = sdt->get_fields();
            const vector<size_t>& metadata_offsets = sdt->get_metadata_offsets();
            size_t *offsets = reinterpret_cast<size_t *>(metadata);
            for (size_t i = 0; i < fields.size(); ++i) {
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
                if (fields[i].extended()) {
                    fill_metadata_from_numpy_dtype(fields[i], fld_dtype, metadata + metadata_offsets[i]);
                }
            }
            break;
        }
        case strided_array_type_id: {
            // The Numpy subarray becomes a series of strided_array_dtypes, so we
            // need to copy the strides into the metadata.
            dtype el;
            PyArray_ArrayDescr *adescr = d->subarray;
            int ndim;
            if (PyTuple_Check(adescr->shape)) {
                ndim = (int)PyTuple_GET_SIZE(adescr->shape);
                strided_array_dtype_metadata *md = reinterpret_cast<strided_array_dtype_metadata *>(metadata);
                intptr_t stride = PyArray_ITEMSIZE(adescr->base);
                el = dt;
                for (int i = ndim-1; i >= 0; --i) {
                    md[i].size = pyobject_as_index(PyTuple_GET_ITEM(adescr->shape, i));
                    md[i].stride = stride;
                    stride *= md[i].size;
                    el = static_cast<const strided_array_dtype *>(el.extended())->get_element_dtype();
                }
                metadata += ndim * sizeof(strided_array_dtype_metadata);
            } else {
                ndim = 1;
                strided_array_dtype_metadata *md = reinterpret_cast<strided_array_dtype_metadata *>(metadata);
                metadata += sizeof(strided_array_dtype_metadata);
                md->size = pyobject_as_index(adescr->shape);
                md->stride = PyArray_ITEMSIZE(adescr->base);
                el = static_cast<const strided_array_dtype *>(dt.extended())->get_element_dtype();
            }
            // Fill the metadata for the array element, if necessary
            if (el.extended()) {
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
                    result->elsize = (int)fdt->get_element_size();
                    return result;
                case string_encoding_utf_32:
                    result = PyArray_DescrNewFromType(NPY_UNICODE);
                    result->elsize = (int)fdt->get_element_size();
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

            pyobject_ownref itemsize_obj(PyLong_FromSize_t(dt.get_element_size()));

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
            const vector<dtype>& fields = sdt->get_fields();
            const vector<string>& field_names = sdt->get_field_names();
            const vector<size_t>& metadata_offsets = sdt->get_metadata_offsets();
            const intptr_t *offsets = reinterpret_cast<const intptr_t *>(metadata);
            int num_fields = (int)fields.size();

            pyobject_ownref names_obj(PyList_New(num_fields));
            for (size_t i = 0; i < num_fields; ++i) {
                PyList_SET_ITEM((PyObject *)names_obj, i, PyString_FromString(field_names[i].c_str()));
            }

            pyobject_ownref formats_obj(PyList_New(num_fields));
            for (size_t i = 0; i < num_fields; ++i) {
                PyList_SET_ITEM((PyObject *)formats_obj, i, (PyObject *)numpy_dtype_from_dtype(fields[i], metadata + metadata_offsets[i]));
            }

            pyobject_ownref offsets_obj(PyList_New(num_fields));
            for (size_t i = 0; i < num_fields; ++i) {
                PyList_SET_ITEM((PyObject *)offsets_obj, i, PyLong_FromSize_t(offsets[i]));
            }

            pyobject_ownref itemsize_obj(PyLong_FromSize_t(dt.get_element_size()));

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
            switch (d.string_encoding()) {
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

// The function ndobject_as_numpy_struct_capsule is exposed even without building against numpy
static void free_array_interface(void *ptr, void *extra_ptr)
{
    PyArrayInterface* inter = (PyArrayInterface *)ptr;
    memory_block_ptr *extra = (memory_block_ptr *)extra_ptr;
    delete[] inter->strides;
    Py_XDECREF(inter->descr);
    delete inter;
    delete extra;
}

static PyObject* struct_ndobject_as_numpy_struct_capsule(const dynd::ndobject& n, int ndim, const intptr_t *shape,
                const intptr_t *strides, const dtype& dt, const char *metadata)
{
    bool writeable = (n.get_access_flags() & write_access_flag) != 0;

    pyobject_ownref descr((PyObject *)numpy_dtype_from_dtype(n.get_dtype(), metadata));

    PyArrayInterface inter;
    memset(&inter, 0, sizeof(inter));

    inter.two = 2;
    inter.nd = n.get_dtype().get_uniform_ndim();
    inter.typekind = 'V';
    inter.itemsize = ((PyArray_Descr *)descr.get())->elsize;
    inter.flags = NPY_ARRAY_ALIGNED | (writeable ? NPY_ARRAY_WRITEABLE : 0);
    if (writeable) {
        inter.data = n.get_readwrite_originptr();
    } else {
        inter.data = const_cast<char *>(n.get_readonly_originptr());
    }
    inter.strides = new intptr_t[2 * inter.nd];
    inter.shape = inter.strides + inter.nd;
    inter.descr = descr.release();

    memcpy(inter.strides, strides, ndim * sizeof(intptr_t));
    memcpy(inter.shape, shape, ndim * sizeof(intptr_t));

    // TODO: Check for Python 3, use PyCapsule there
    return PyCObject_FromVoidPtrAndDesc(new PyArrayInterface(inter), new memory_block_ptr(n.get_memblock()), free_array_interface);
}

PyObject* pydynd::ndobject_as_numpy_struct_capsule(const dynd::ndobject& n)
{
    dtype dt = n.get_dtype();
    int ndim = dt.get_uniform_ndim();
    dimvector shape(ndim), strides(ndim);
    const char *metadata = n.get_ndo_meta();
    // Follow the chain of strided array dtypes to convert them into numpy shape/strides
    for (int i = 0; i < ndim; ++i) {
        if (dt.get_type_id() != strided_array_type_id) {
            stringstream ss;
            ss << "Cannot view ndobject with dtype " << n.get_dtype() << " directly as a numpy array";
            throw runtime_error(ss.str());
        }
        const strided_array_dtype_metadata *md = reinterpret_cast<const strided_array_dtype_metadata *>(metadata);
        shape[i] = md->size;
        strides[i] = md->stride;
        metadata += sizeof(strided_array_dtype_metadata);
        dt = static_cast<const strided_array_dtype *>(dt.extended())->get_element_dtype();
    }

    if (dt.get_type_id() == struct_type_id) {
        return struct_ndobject_as_numpy_struct_capsule(n, ndim, shape.get(), strides.get(), dt, metadata);
    }

    bool byteswapped = false;
    if (dt.get_type_id() == byteswap_type_id) {
        dt = dt.operand_dtype();
        byteswapped = true;
    }

    bool aligned = true;
    if (dt.get_type_id() == view_type_id) {
        dtype sdt = dt.operand_dtype();
        if (sdt.get_type_id() == fixedbytes_type_id) {
            dt = dt.value_dtype();
            aligned = false;
        }
    }

    bool writeable = (n.get_access_flags() & write_access_flag) != 0;

    PyArrayInterface inter;
    memset(&inter, 0, sizeof(inter));

    inter.two = 2;
    inter.nd = ndim;
    inter.typekind = numpy_kindchar_of(dt);
    // Numpy treats 'U' as number of 4-byte characters, not number of bytes
    inter.itemsize = (int)(inter.typekind != 'U' ? dt.get_element_size() : dt.get_element_size() / 4);
    inter.flags = (byteswapped ? 0 : NPY_ARRAY_NOTSWAPPED) |
                  (aligned ? NPY_ARRAY_ALIGNED : 0) |
                  (writeable ? NPY_ARRAY_WRITEABLE : 0);
    if (writeable) {
        inter.data = n.get_readwrite_originptr();
    } else {
        inter.data = const_cast<char *>(n.get_readonly_originptr());
    }
    inter.strides = new intptr_t[2 * ndim];
    inter.shape = inter.strides + ndim;

    memcpy(inter.strides, strides.get(), ndim * sizeof(intptr_t));
    memcpy(inter.shape, shape.get(), ndim * sizeof(intptr_t));

    // TODO: Check for Python 3, use PyCapsule there
    return PyCObject_FromVoidPtrAndDesc(new PyArrayInterface(inter), new memory_block_ptr(n.get_data_memblock()), free_array_interface);
}

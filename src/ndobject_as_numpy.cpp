//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "numpy_interop.hpp"

#if DYND_NUMPY_INTEROP

#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>

#include "ndobject_as_numpy.hpp"
#include "numpy_interop.hpp"
#include "ndobject_functions.hpp"
#include "utility_functions.hpp"

#include <dynd/dtypes/strided_dim_dtype.hpp>
#include <dynd/dtypes/fixed_dim_dtype.hpp>
#include <dynd/dtypes/fixedstring_dtype.hpp>
#include <dynd/dtypes/base_struct_dtype.hpp>
#include <dynd/dtypes/date_dtype.hpp>
#include <dynd/dtypes/bytes_dtype.hpp>
#include <dynd/dtypes/property_dtype.hpp>
#include <dynd/shape_tools.hpp>

using namespace std;
using namespace dynd;
using namespace pydynd;

static int dynd_to_numpy_type_id[builtin_type_id_count] = {
    NPY_NOTYPE,
    NPY_BOOL,
    NPY_INT8,
    NPY_INT16,
    NPY_INT32,
    NPY_INT64,
    NPY_UINT8,
    NPY_UINT16,
    NPY_UINT32,
    NPY_UINT64,
    NPY_FLOAT32,
    NPY_FLOAT64,
    NPY_COMPLEX64,
    NPY_COMPLEX128,
    NPY_NOTYPE
};

static void make_numpy_dtype_for_copy(pyobject_ownref *out_numpy_dtype, 
                size_t undim, const dtype& dt, const char *metadata)
{
    // DyND builtin types
    if (dt.is_builtin()) {
        out_numpy_dtype->reset((PyObject *)PyArray_DescrFromType(
                        dynd_to_numpy_type_id[dt.get_type_id()]));
        return;
    }

    switch (dt.get_type_id()) {
            case fixedstring_type_id: {
                const fixedstring_dtype *fsd = static_cast<const fixedstring_dtype *>(dt.extended());
                PyArray_Descr *result;
                switch (fsd->get_encoding()) {
                    case string_encoding_ascii:
                        result = PyArray_DescrNewFromType(NPY_STRING);
                        result->elsize = (int)fsd->get_data_size();
                        out_numpy_dtype->reset((PyObject *)result);
                        return;
                    case string_encoding_utf_32:
                        result = PyArray_DescrNewFromType(NPY_UNICODE);
                        result->elsize = (int)fsd->get_data_size();
                        out_numpy_dtype->reset((PyObject *)result);
                        return;
                    default:
                        // If it's not one of the encodings NumPy supports,
                        // use Unicode
                        result = PyArray_DescrNewFromType(NPY_UNICODE);
                        result->elsize = (int)fsd->get_data_size() *
                                        4 / string_encoding_char_size_table[fsd->get_encoding()];
                        out_numpy_dtype->reset((PyObject *)result);
                        return;
                }
                break;
            }
            case date_type_id: {
#if NPY_API_VERSION >= 6 // At least NumPy 1.6
                PyArray_Descr *datedt = NULL;
#  if PY_VERSION_HEX >= 0x03000000
                pyobject_ownref M8str(PyUnicode_FromString("M8[D]"));
#  else
                pyobject_ownref M8str(PyString_FromString("M8[D]"));
#  endif
                if (!PyArray_DescrConverter(M8str.get(), &datedt)) {
                    throw runtime_error("Failed to create NumPy datetime64[D] dtype");
                }
                out_numpy_dtype->reset((PyObject *)datedt);
                return;
#else
                throw runtime_error("NumPy >= 1.6 is required for dynd date dtype interop");
#endif
            }
        case strided_dim_type_id:
        case fixed_dim_type_id: {
            if (undim > 0) {
                // If this is one of the uniform dimensions, it simply
                // becomes one of the numpy ndarray dimensions
                if (dt.get_type_id() == strided_dim_type_id) {
                    const strided_dim_dtype *sad = static_cast<const strided_dim_dtype *>(dt.extended());
                    make_numpy_dtype_for_copy(out_numpy_dtype,
                                    undim - 1, sad->get_element_dtype(),
                                    metadata + sizeof(strided_dim_dtype_metadata));
                } else {
                    const fixed_dim_dtype *fad = static_cast<const fixed_dim_dtype *>(dt.extended());
                    make_numpy_dtype_for_copy(out_numpy_dtype,
                                    undim - 1, fad->get_element_dtype(),
                                    metadata + sizeof(strided_dim_dtype_metadata));
                }
                return;
            } else {
                // If this isn't one of the uniform dimensions, it maps into
                // a NumPy dtype with a shape
                if (metadata == NULL) {
                    stringstream ss;
                    ss << "cannot determine shape of dynd type " << dt;
                    ss << " to convert it to a numpy dtype";
                    throw runtime_error(ss.str());
                }
                // Build up the shape of the array for NumPy
                pyobject_ownref shape(PyList_New(0));
                dtype element_dtype = dt;
                while(undim > 0) {
                    size_t dim_size = 0;
                    if (dt.get_type_id() == strided_dim_type_id) {
                        const strided_dim_dtype *sad =
                                        static_cast<const strided_dim_dtype *>(element_dtype.extended());
                        dim_size = sad->get_dim_size(metadata, NULL);
                        element_dtype = sad->get_element_dtype();
                        metadata += sizeof(strided_dim_dtype_metadata);
                    } else if (dt.get_type_id() == fixed_dim_type_id) {
                        const fixed_dim_dtype *fad =
                                        static_cast<const fixed_dim_dtype *>(element_dtype.extended());
                        dim_size = fad->get_fixed_dim_size();
                        element_dtype = fad->get_element_dtype();
                    } else {
                        stringstream ss;
                        ss << "dynd as_numpy could not convert dynd type ";
                        ss << dt;
                        ss << " to a numpy dtype";
                        throw runtime_error(ss.str());
                    }
                    --undim;
                    if (PyList_Append(shape.get(), PyLong_FromSize_t(dim_size)) < 0) {
                        throw runtime_error("propagating python error");
                    }
                }
                // Get the NumPy dtype of the element
                pyobject_ownref child_numpy_dtype;
                make_numpy_dtype_for_copy(&child_numpy_dtype,
                                0, element_dtype, metadata);
                // Create the result NumPy dtype
                pyobject_ownref tuple_obj(PyTuple_New(2));
                PyTuple_SET_ITEM(tuple_obj.get(), 0, child_numpy_dtype.release());
                PyTuple_SET_ITEM(tuple_obj.get(), 1, shape.release());

                PyArray_Descr *result = NULL;
                if (!PyArray_DescrConverter(tuple_obj, &result)) {
                    throw runtime_error("failed to convert dynd dtype into numpy subarray dtype");
                }
                // Put the final dtype reference in the output
                out_numpy_dtype->reset((PyObject *)result);
                return;
            }
            break;
        }
        case cstruct_type_id:
        case struct_type_id: {
            const base_struct_dtype *bs = static_cast<const base_struct_dtype *>(dt.extended());
            const dtype *field_types = bs->get_field_types();
            const string *field_names = bs->get_field_names();
            size_t field_count = bs->get_field_count();

            pyobject_ownref names_obj(PyList_New(field_count));
            for (size_t i = 0; i < field_count; ++i) {
#if PY_VERSION_HEX >= 0x03000000
                pyobject_ownref name_str(PyUnicode_FromStringAndSize(
                                field_names[i].data(), field_names[i].size()));
#else
                pyobject_ownref name_str(PyString_FromStringAndSize(
                                field_names[i].data(), field_names[i].size()));
#endif
                PyList_SET_ITEM(names_obj.get(), i, name_str.release());
            }

            pyobject_ownref formats_obj(PyList_New(field_count));
            pyobject_ownref offsets_obj(PyList_New(field_count));
            size_t standard_offset = 0, standard_alignment = 1;
            for (size_t i = 0; i < field_count; ++i) {
                // Get the NumPy dtype of the element
                pyobject_ownref field_numpy_dtype;
                make_numpy_dtype_for_copy(&field_numpy_dtype,
                                0, field_types[i], metadata);
                size_t field_alignment = ((PyArray_Descr *)field_numpy_dtype.get())->alignment;
                size_t field_size = ((PyArray_Descr *)field_numpy_dtype.get())->elsize;
                standard_offset = inc_to_alignment(standard_offset, field_alignment);
                standard_alignment = max(standard_alignment, field_alignment);
                PyList_SET_ITEM(formats_obj.get(), i, field_numpy_dtype.release());
                PyList_SET_ITEM((PyObject *)offsets_obj, i, PyLong_FromSize_t(standard_offset));
                standard_offset += field_size;
            }
            // Get the full element size
            standard_offset = inc_to_alignment(standard_offset, standard_alignment);
            pyobject_ownref itemsize_obj(PyLong_FromSize_t(standard_offset));

            pyobject_ownref dict_obj(PyDict_New());
            PyDict_SetItemString(dict_obj, "names", names_obj);
            PyDict_SetItemString(dict_obj, "formats", formats_obj);
            PyDict_SetItemString(dict_obj, "offsets", offsets_obj);
            PyDict_SetItemString(dict_obj, "itemsize", itemsize_obj);

            PyArray_Descr *result = NULL;
            if (!PyArray_DescrAlignConverter(dict_obj, &result)) {
                stringstream ss;
                ss << "failed to convert dtype " << dt << " into numpy dtype via dict";
                throw runtime_error(ss.str());
            }
            out_numpy_dtype->reset((PyObject *)result);
            return;
        }
        default: {
            break;
        }
    }

    if (dt.get_kind() == expression_kind) {
        // Convert the value dtype for the copy
        make_numpy_dtype_for_copy(out_numpy_dtype, undim,
                        dt.value_dtype(), NULL);
        return;
    }

    // Anything which fell through is an error
    stringstream ss;
    ss << "dynd as_numpy could not convert dynd type ";
    ss << dt;
    ss << " to a numpy dtype";
    throw runtime_error(ss.str());
}

static void as_numpy_analysis(pyobject_ownref *out_numpy_dtype, bool *out_requires_copy,
                size_t undim, const dtype& dt, const char *metadata)
{
    // DyND builtin types
    if (dt.is_builtin()) {
        out_numpy_dtype->reset((PyObject *)PyArray_DescrFromType(
                        dynd_to_numpy_type_id[dt.get_type_id()]));
        return;
    }

    switch (dt.get_type_id()) {
            case fixedstring_type_id: {
                const fixedstring_dtype *fsd = static_cast<const fixedstring_dtype *>(dt.extended());
                PyArray_Descr *result;
                switch (fsd->get_encoding()) {
                    case string_encoding_ascii:
                        result = PyArray_DescrNewFromType(NPY_STRING);
                        result->elsize = (int)fsd->get_data_size();
                        out_numpy_dtype->reset((PyObject *)result);
                        return;
                    case string_encoding_utf_32:
                        result = PyArray_DescrNewFromType(NPY_UNICODE);
                        result->elsize = (int)fsd->get_data_size();
                        out_numpy_dtype->reset((PyObject *)result);
                        return;
                    default:
                        out_numpy_dtype->clear();
                        *out_requires_copy = true;
                        return;
                }
                break;
            }
            case date_type_id: {
#if NPY_API_VERSION >= 6 // At least NumPy 1.6
                out_numpy_dtype->clear();
                *out_requires_copy = true;
                return;
#else
                throw runtime_error("NumPy >= 1.6 is required for dynd date dtype interop");
#endif
            }
        case property_type_id: {
            const property_dtype *pd = static_cast<const property_dtype *>(dt.extended());
            // Special-case of 'int64 as date' property dtype, which is binary
            // compatible with NumPy's "M8[D]"
            if (pd->is_reversed_property() && pd->get_value_dtype().get_type_id() == date_type_id &&
                            pd->get_operand_dtype().get_type_id() == int64_type_id) {
                PyArray_Descr *datedt = NULL;
#  if PY_VERSION_HEX >= 0x03000000
                pyobject_ownref M8str(PyUnicode_FromString("M8[D]"));
#  else
                pyobject_ownref M8str(PyString_FromString("M8[D]"));
#  endif
                if (!PyArray_DescrConverter(M8str.get(), &datedt)) {
                    throw runtime_error("Failed to create NumPy datetime64[D] dtype");
                }
                out_numpy_dtype->reset((PyObject *)datedt);
                return;
            }
            break;
        }
        case byteswap_type_id: {
            const base_expression_dtype *bed = static_cast<const base_expression_dtype *>(dt.extended());
            // Analyze the unswapped version
            as_numpy_analysis(out_numpy_dtype, out_requires_copy,
                            undim, bed->get_value_dtype(), metadata);
            pyobject_ownref swapdt(out_numpy_dtype->release());
            // Byteswap the numpy dtype
            out_numpy_dtype->reset((PyObject *)PyArray_DescrNewByteorder(
                            (PyArray_Descr *)swapdt.get(), NPY_SWAP));
            return;
        }
        case strided_dim_type_id: {
            const strided_dim_dtype *sad = static_cast<const strided_dim_dtype *>(dt.extended());
            if (undim > 0) {
                // If this is one of the uniform dimensions, it simply
                // becomes one of the numpy ndarray dimensions
                as_numpy_analysis(out_numpy_dtype, out_requires_copy,
                                undim - 1, sad->get_element_dtype(),
                                metadata + sizeof(strided_dim_dtype_metadata));
                return;
            } else {
                // If this isn't one of the uniform dimensions, it maps into
                // a NumPy dtype with a shape
                out_numpy_dtype->clear();
                *out_requires_copy = true;
                return;
            }
            break;
        }
        case fixed_dim_type_id: {
            const fixed_dim_dtype *fad = static_cast<const fixed_dim_dtype *>(dt.extended());
            if (undim > 0) {
                // If this is one of the uniform dimensions, it simply
                // becomes one of the numpy ndarray dimensions
                as_numpy_analysis(out_numpy_dtype, out_requires_copy,
                                undim - 1, fad->get_element_dtype(),
                                metadata + sizeof(strided_dim_dtype_metadata));
                return;
            } else {
                // If this isn't one of the uniform dimensions, it maps into
                // a NumPy dtype with a shape
                // Build up the shape of the array for NumPy
                pyobject_ownref shape(PyList_New(0));
                dtype element_dtype = dt;
                while(undim > 0) {
                    size_t dim_size = 0;
                    if (dt.get_type_id() == fixed_dim_type_id) {
                        const fixed_dim_dtype *fad =
                                        static_cast<const fixed_dim_dtype *>(element_dtype.extended());
                        element_dtype = fad->get_element_dtype();
                        if (fad->get_data_size() !=
                                        element_dtype.get_data_size() * dim_size) {
                            // If it's not C-order, a copy is required
                            out_numpy_dtype->clear();
                            *out_requires_copy = true;
                            return;
                        }
                    } else {
                        stringstream ss;
                        ss << "dynd as_numpy could not convert dynd type ";
                        ss << dt;
                        ss << " to a numpy dtype";
                        throw runtime_error(ss.str());
                    }
                    --undim;
                    if (PyList_Append(shape.get(), PyLong_FromSize_t(dim_size)) < 0) {
                        throw runtime_error("propagating python error");
                    }
                }
                // Get the NumPy dtype of the element
                pyobject_ownref child_numpy_dtype;
                as_numpy_analysis(&child_numpy_dtype, out_requires_copy,
                                0, element_dtype, metadata);
                if (*out_requires_copy) {
                    // If the child required a copy, stop right away
                    out_numpy_dtype->clear();
                    return;
                }
                // Create the result NumPy dtype
                pyobject_ownref tuple_obj(PyTuple_New(2));
                PyTuple_SET_ITEM(tuple_obj.get(), 0, child_numpy_dtype.release());
                PyTuple_SET_ITEM(tuple_obj.get(), 1, shape.release());

                PyArray_Descr *result = NULL;
                if (!PyArray_DescrConverter(tuple_obj, &result)) {
                    throw runtime_error("failed to convert dynd dtype into numpy subarray dtype");
                }
                // Put the final dtype reference in the output
                out_numpy_dtype->reset((PyObject *)result);
                return;
            }
            break;
        }
        case cstruct_type_id:
        case struct_type_id: {
            if (dt.get_type_id() == struct_type_id && metadata == NULL) {
                // If it's a struct dtype with no metadata, a copy is required
                out_numpy_dtype->clear();
                *out_requires_copy = true;
                return;
            }
            const base_struct_dtype *bs = static_cast<const base_struct_dtype *>(dt.extended());
            const dtype *field_types = bs->get_field_types();
            const string *field_names = bs->get_field_names();
            const size_t *offsets = bs->get_data_offsets(metadata);
            size_t field_count = bs->get_field_count();

            pyobject_ownref names_obj(PyList_New(field_count));
            for (size_t i = 0; i < field_count; ++i) {
#if PY_VERSION_HEX >= 0x03000000
                pyobject_ownref name_str(PyUnicode_FromStringAndSize(
                                field_names[i].data(), field_names[i].size()));
#else
                pyobject_ownref name_str(PyString_FromStringAndSize(
                                field_names[i].data(), field_names[i].size()));
#endif
                PyList_SET_ITEM(names_obj.get(), i, name_str.release());
            }

            pyobject_ownref formats_obj(PyList_New(field_count));
            for (size_t i = 0; i < field_count; ++i) {
                // Get the NumPy dtype of the element
                pyobject_ownref field_numpy_dtype;
                as_numpy_analysis(&field_numpy_dtype, out_requires_copy,
                                0, field_types[i], metadata);
                if (*out_requires_copy) {
                    // If the field required a copy, stop right away
                    out_numpy_dtype->clear();
                    return;
                }
                PyList_SET_ITEM(formats_obj.get(), i, field_numpy_dtype.release());
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
            if (!PyArray_DescrConverter(dict_obj, &result)) {
                stringstream ss;
                ss << "failed to convert dtype " << dt << " into numpy dtype via dict";
                throw runtime_error(ss.str());
            }
            out_numpy_dtype->reset((PyObject *)result);
            return;
        }
        default: {
            break;
        }
    }

    if (dt.get_kind() == expression_kind) {
        // If none of the prior checks caught this expression,
        // a copy is required.
        out_numpy_dtype->clear();
        *out_requires_copy = true;
        return;
    }

    // Anything which fell through is an error
    stringstream ss;
    ss << "dynd as_numpy could not convert dynd type ";
    ss << dt;
    ss << " to a numpy dtype";
    throw runtime_error(ss.str());
}


PyObject *pydynd::ndobject_as_numpy(PyObject *n_obj, bool allow_copy)
{
    if (!WNDObject_Check(n_obj)) {
        throw runtime_error("can only call dynd's as_numpy on dynd ndobjects");
    }
    ndobject n = ((WNDObject *)n_obj)->v;
    if (n.get_ndo() == NULL) {
        throw runtime_error("cannot convert NULL dynd ndobject to numpy");
    }

    // If a copy is allowed, convert the builtin scalars to NumPy scalars
    if (allow_copy && n.get_dtype().is_scalar()) {
        pyobject_ownref result;
        switch (n.get_dtype().get_type_id()) {
            case uninitialized_type_id:
                throw runtime_error("cannot convert uninitialized dynd ndobject to numpy");
            case void_type_id:
                throw runtime_error("cannot convert void dynd ndobject to numpy");
            case bool_type_id:
                if (*n.get_readonly_originptr()) {
                    Py_INCREF(PyArrayScalar_True);
                    result.reset(PyArrayScalar_True);
                } else {
                    Py_INCREF(PyArrayScalar_False);
                    result.reset(PyArrayScalar_False);
                }
                break;
            case int8_type_id:
                result.reset(PyArrayScalar_New(Int8));
                PyArrayScalar_ASSIGN(result.get(), Int8,
                                *reinterpret_cast<const int8_t *>(n.get_readonly_originptr()));
                break;
            case int16_type_id:
                result.reset(PyArrayScalar_New(Int16));
                PyArrayScalar_ASSIGN(result.get(), Int16,
                                *reinterpret_cast<const int16_t *>(n.get_readonly_originptr()));
                break;
            case int32_type_id:
                result.reset(PyArrayScalar_New(Int32));
                PyArrayScalar_ASSIGN(result.get(), Int32,
                                *reinterpret_cast<const int32_t *>(n.get_readonly_originptr()));
                break;
            case int64_type_id:
                result.reset(PyArrayScalar_New(Int64));
                PyArrayScalar_ASSIGN(result.get(), Int64,
                                *reinterpret_cast<const int64_t *>(n.get_readonly_originptr()));
                break;
            case uint8_type_id:
                result.reset(PyArrayScalar_New(UInt8));
                PyArrayScalar_ASSIGN(result.get(), UInt8,
                                *reinterpret_cast<const uint8_t *>(n.get_readonly_originptr()));
                break;
            case uint16_type_id:
                result.reset(PyArrayScalar_New(UInt16));
                PyArrayScalar_ASSIGN(result.get(), UInt16,
                                *reinterpret_cast<const uint16_t *>(n.get_readonly_originptr()));
                break;
            case uint32_type_id:
                result.reset(PyArrayScalar_New(UInt32));
                PyArrayScalar_ASSIGN(result.get(), UInt32,
                                *reinterpret_cast<const uint32_t *>(n.get_readonly_originptr()));
                break;
            case uint64_type_id:
                result.reset(PyArrayScalar_New(UInt64));
                PyArrayScalar_ASSIGN(result.get(), UInt64,
                                *reinterpret_cast<const uint64_t *>(n.get_readonly_originptr()));
                break;
            case float32_type_id:
                result.reset(PyArrayScalar_New(Float32));
                PyArrayScalar_ASSIGN(result.get(), Float32,
                                *reinterpret_cast<const float *>(n.get_readonly_originptr()));
                break;
            case float64_type_id:
                result.reset(PyArrayScalar_New(Float64));
                PyArrayScalar_ASSIGN(result.get(), Float64,
                                *reinterpret_cast<const double *>(n.get_readonly_originptr()));
                break;
            case complex_float32_type_id:
                result.reset(PyArrayScalar_New(Complex64));
                PyArrayScalar_VAL(result.get(), Complex64).real = 
                                reinterpret_cast<const float *>(n.get_readonly_originptr())[0];
                PyArrayScalar_VAL(result.get(), Complex64).imag = 
                                reinterpret_cast<const float *>(n.get_readonly_originptr())[1];
                break;
            case complex_float64_type_id:
                result.reset(PyArrayScalar_New(Complex128));
                PyArrayScalar_VAL(result.get(), Complex128).real = 
                                reinterpret_cast<const double *>(n.get_readonly_originptr())[0];
                PyArrayScalar_VAL(result.get(), Complex128).imag = 
                                reinterpret_cast<const double *>(n.get_readonly_originptr())[1];
                break;
            case date_type_id: {
#if NPY_API_VERSION >= 6 // At least NumPy 1.6
                int32_t dateval = *reinterpret_cast<const int32_t *>(n.get_readonly_originptr());
                result.reset(PyArrayScalar_New(Datetime));

                PyArrayScalar_VAL(result.get(), Datetime) =
                                (dateval == DYND_DATE_NA) ? NPY_DATETIME_NAT
                                                          : dateval;
                PyArray_DatetimeMetaData &meta = ((PyDatetimeScalarObject *)result.get())->obmeta;
                meta.base = NPY_FR_D;
                meta.num = 1;
#  if NPY_API_VERSION == 6 // Only for NumPy 1.6
                meta.den = 1;
                meta.events = 1;
#  endif
                break;
#else
                throw runtime_error("NumPy >= 1.6 is required for dynd date dtype interop");
#endif
            }
            default: {
                // Because 'allow_copy' is true
                // we can evaluate any expressions and
                // make copies of strings
                if (n.get_dtype().get_kind() == expression_kind) {
                    // If it's an expression kind
                    pyobject_ownref n_tmp(wrap_ndobject(n.eval()));
                    return ndobject_as_numpy(n_tmp.get(), true);
                } else if (n.get_dtype().get_kind() == string_kind) {
                    // If it's a string kind, return it as a Python unicode
                    return ndobject_as_py(n);
                }
                stringstream ss;
                ss << "dynd as_numpy could not convert dynd type ";
                ss << n.get_dtype();
                ss << " to a numpy dtype";
                throw runtime_error(ss.str());
            }
        }
        return result.release();
    }

    if (n.get_dtype().get_type_id() == pointer_type_id ||
                    n.get_dtype().get_type_id() == var_dim_type_id) {
        // If it's a pointer or var_dim, use 0-length indexing to
        // strip away this leading part so it's compatible with NumPy.
        pyobject_ownref n_tmp(wrap_ndobject(n.at_array(0, NULL)));
        return ndobject_as_numpy(n_tmp.get(), allow_copy);
    }

    // Do a recursive analysis of the ndobject for how to
    // convert it to NumPy
    bool requires_copy = false;
    pyobject_ownref numpy_dtype;
    size_t undim = n.get_undim();
    dimvector shape(undim), strides(undim);

    n.get_shape(shape.get());
    n.get_strides(strides.get());
    as_numpy_analysis(&numpy_dtype, &requires_copy,
                    undim, n.get_dtype(), n.get_ndo_meta());
    if (requires_copy) {
        if (!allow_copy) {
            stringstream ss;
            ss << "cannot view dynd object with dtype " << n.get_dtype();
            ss << " as numpy without making a copy";
            throw runtime_error(ss.str());
        }
        make_numpy_dtype_for_copy(&numpy_dtype,
                        undim, n.get_dtype(), n.get_ndo_meta());

        // Rebuild the strides so that the copy follows 'KEEPORDER'
        intptr_t element_size = ((PyArray_Descr *)numpy_dtype.get())->elsize;
        if (undim == 1) {
            strides[0] = element_size;
        } else if (undim > 1) {
            shortvector<int> axis_perm(undim);
            strides_to_axis_perm(undim, strides.get(), axis_perm.get());
            axis_perm_to_strides(undim, axis_perm.get(),
                            shape.get(), element_size,
                            strides.get());
        }

        // Create a new NumPy array, and copy from the dynd ndobject
        pyobject_ownref result(PyArray_NewFromDescr(&PyArray_Type, (PyArray_Descr *)numpy_dtype.release(),
                        (int)undim, shape.get(), strides.get(), NULL, 0, NULL));
        // Create a dynd ndobject view of this result
        ndobject result_dynd = ndobject_from_numpy_array((PyArrayObject *)result.get());
        // Copy the values using this view
        result_dynd.vals() = n;
        // Return the NumPy array
        return result.release();
    } else {
        // Create a view directly to the dynd ndobject
        pyobject_ownref result(PyArray_NewFromDescr(&PyArray_Type, (PyArray_Descr *)numpy_dtype.release(),
                    (int)undim, shape.get(), strides.get(), n.get_ndo()->m_data_pointer,
                    ((n.get_flags()&write_access_flag) ? NPY_ARRAY_WRITEABLE : 0) | NPY_ARRAY_ALIGNED, NULL));
#if NPY_API_VERSION >= 7 // At least NumPy 1.7
        Py_INCREF(n_obj);
        if (PyArray_SetBaseObject((PyArrayObject *)result.get(), n_obj) < 0) {
            throw runtime_error("propagating python exception");
        }
#else
        PyArray_BASE(result.get()) = n_obj;
        Py_INCREF(n_obj);
#endif
        return result.release();
    }
}

#endif // NUMPY_INTEROP


//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <dynd/dtypes/strided_array_dtype.hpp>
#include <dynd/dtypes/fixedarray_dtype.hpp>
#include <dynd/dtypes/fixedstruct_dtype.hpp>
#include <dynd/dtypes/fixedstring_dtype.hpp>
#include <dynd/shape_tools.hpp>

#include "ndobject_as_pep3118.hpp"
#include "ndobject_functions.hpp"
#include "utility_functions.hpp"

using namespace std;
using namespace dynd;
using namespace pydynd;

static void append_pep3118_format(intptr_t& out_itemsize, const dtype& dt, const char *metadata, std::stringstream& o)
{
    switch (dt.get_type_id()) {
        case bool_type_id:
            o << "?";
            out_itemsize = 1;
            return;
        case int8_type_id:
            o << "b";
            out_itemsize = 1;
            return;
        case int16_type_id:
            o << "h";
            out_itemsize = 2;
            return;
        case int32_type_id:
            o << "i";
            out_itemsize = 4;
            return;
        case int64_type_id:
            o << "q";
            out_itemsize = 8;
            return;
        case uint8_type_id:
            o << "B";
            out_itemsize = 1;
            return;
        case uint16_type_id:
            o << "H";
            out_itemsize = 2;
            return;
        case uint32_type_id:
            o << "I";
            out_itemsize = 4;
            return;
        case uint64_type_id:
            o << "Q";
            out_itemsize = 8;
            return;
        case float32_type_id:
            o << "f";
            out_itemsize = 4;
            return;
        case float64_type_id:
            o << "d";
            out_itemsize = 8;
            return;
        case complex_float32_type_id:
            o << "Zf";
            out_itemsize = 8;
            return;
        case complex_float64_type_id:
            o << "Zd";
            out_itemsize = 16;
            return;
        case fixedstring_type_id:
            switch (static_cast<const fixedstring_dtype *>(dt.extended())->get_encoding()) {
                case string_encoding_ascii: {
                    intptr_t element_size = dt.get_element_size();
                    o << element_size << "s";
                    out_itemsize = element_size;
                    return;
                }
                // TODO: Couldn't find documentation for UCS-2 character code?
                case string_encoding_utf_32:
                    intptr_t element_size = dt.get_element_size();
                    o << (element_size/4) << "w";
                    out_itemsize = element_size;
                    return;
            }
            // Pass through to error
            break;
        case fixedarray_type_id: {
            dtype child_dt = dt;
            o << "(";
            do {
                const fixedarray_dtype *tdt = static_cast<const fixedarray_dtype *>(child_dt.extended());
                size_t dim_size = tdt->get_fixed_dim_size();
                o << dim_size;
                if (child_dt.get_element_size() != tdt->get_element_dtype().get_element_size() * dim_size) {
                    stringstream ss;
                    ss << "Cannot convert dynd dtype " << dt << " into a PEP 3118 format because it is not C-order";
                    throw runtime_error(ss.str());
                }
                o << ")";
                child_dt = tdt->get_element_dtype();
            } while (child_dt.get_type_id() == fixedarray_type_id && (o << ","));
            append_pep3118_format(out_itemsize, child_dt, metadata, o);
            out_itemsize = dt.get_element_size();
            return;
        }
        case fixedstruct_type_id: {
            o << "T{";
            const fixedstruct_dtype *tdt = static_cast<const fixedstruct_dtype *>(dt.extended());
            const vector<dtype>& field_types = tdt->get_field_types();
            const vector<string>& field_names = tdt->get_field_names();
            size_t num_fields = field_types.size();
            const vector<size_t>& offsets = tdt->get_data_offsets();
            const vector<size_t>& metadata_offsets = tdt->get_metadata_offsets();
            size_t format_offset = 0;
            for (size_t i = 0; i != num_fields; ++i) {
                size_t offset = offsets[i];
                // Add padding bytes
                while (offset > format_offset) {
                    o << "x";
                    ++format_offset;
                }
                // The field's dtype
                append_pep3118_format(out_itemsize, field_types[i], metadata ? (metadata + metadata_offsets[i]) : NULL, o);
                format_offset += out_itemsize;
                // Append the name
                o << ":" << field_names[i] << ":";
            }
            out_itemsize = dt.get_element_size();
            // Add padding bytes to the end
            while ((size_t)out_itemsize > format_offset) {
                o << "x";
                ++format_offset;
            }
            o << "}";
            return;
        }
        default:
            break;
    }
    stringstream ss;
    ss << "Cannot convert dynd dtype " << dt << " into a PEP 3118 format string";
    throw runtime_error(ss.str());
}

std::string pydynd::make_pep3118_format(intptr_t& out_itemsize, const dtype& dt, const char *metadata)
{
    std::stringstream result;
    // Specify native alignment/storage if it's a builtin scalar type
    if (dt.extended() == NULL) {
        result << "@";
    }
    append_pep3118_format(out_itemsize, dt, metadata, result);
    return result.str();
}

static void ndobject_getbuffer_pep3118_bytes(const dtype& dt, const char *metadata, char *data, Py_buffer *buffer, int flags)
{
    buffer->itemsize = 1;
    if (flags&PyBUF_FORMAT) {
        buffer->format = (char *)"c";
    } else {
        buffer->format = NULL;
    }
    buffer->ndim = 1;
    buffer->shape = &buffer->smalltable[0];
    buffer->strides = &buffer->smalltable[1];
    buffer->strides[0] = 1;

    if (dt.get_type_id() == bytes_type_id) {
        // Variable-length bytes type
        char **bytes_data = reinterpret_cast<char **>(data);
        buffer->buf = bytes_data[0];
        buffer->len = (bytes_data[1] - bytes_data[0]);
    } else {
        // Fixed-length bytes type
        buffer->len = dt.get_element_size();
    }
    buffer->shape[0] = buffer->len;
}

int pydynd::ndobject_getbuffer_pep3118(PyObject *ndo, Py_buffer *buffer, int flags)
{
    try {
        buffer->shape = NULL;
        buffer->strides = NULL;
        buffer->suboffsets = NULL;
        buffer->format = NULL;
        buffer->obj = ndo;
        buffer->internal = NULL;
        Py_INCREF(ndo);
        if (!WNDObject_Check(ndo)) {
            throw runtime_error("ndobject_getbuffer_pep3118 called on a non-ndobject");
        }
        ndobject& n = ((WNDObject *)ndo)->v;
        ndobject_preamble *preamble = n.get_ndo();
        dtype dt = n.get_dtype();

        // Check if a writeable buffer is requested
        if ((flags&PyBUF_WRITABLE) && !(n.get_access_flags()&write_access_flag)) {
            throw runtime_error("dynd ndobject is not writeable");
        }
        buffer->readonly = ((n.get_access_flags()&write_access_flag) == 0);
        buffer->buf = preamble->m_data_pointer;

        if (dt.get_type_id() == bytes_type_id || dt.get_type_id() == fixedbytes_type_id) {
            ndobject_getbuffer_pep3118_bytes(dt, n.get_ndo_meta(), n.get_ndo()->m_data_pointer, buffer, flags);
            return 0;
        }

        buffer->ndim = dt.get_uniform_ndim();
        if ((flags&PyBUF_ND) && buffer->ndim > 1) {
            stringstream ss;
            ss << "dynd dtype " << n.get_dtype() << " is multidimensional, but PEP 3118 request is not ND";
            throw runtime_error(ss.str());
        }

        // Create the format, and allocate the dynamic memory but Py_buffer needs
        char *uniform_metadata = n.get_ndo_meta();
        dtype uniform_dtype = dt.get_dtype_at_dimension(&uniform_metadata, buffer->ndim);
        if ((flags&PyBUF_FORMAT) || uniform_dtype.get_element_size() == 0) {
            string format = make_pep3118_format(buffer->itemsize, uniform_dtype, uniform_metadata);
            if (flags&PyBUF_FORMAT) {
                buffer->internal = malloc(2*buffer->ndim*sizeof(intptr_t) + format.size() + 1);
                buffer->shape = reinterpret_cast<Py_ssize_t *>(buffer->internal);
                buffer->strides = buffer->shape + buffer->ndim;
                buffer->format = reinterpret_cast<char *>(buffer->strides + buffer->ndim);
                memcpy(buffer->format, format.c_str(), format.size() + 1);
            } else {
                buffer->format = NULL;
                buffer->internal = malloc(2*buffer->ndim*sizeof(intptr_t));
                buffer->shape = reinterpret_cast<Py_ssize_t *>(buffer->internal);
                buffer->strides = buffer->shape + buffer->ndim;
            }
        } else {
            buffer->format = NULL;
            buffer->itemsize = uniform_dtype.get_element_size();
            buffer->internal = malloc(2*buffer->ndim*sizeof(intptr_t));
            buffer->shape = reinterpret_cast<Py_ssize_t *>(buffer->internal);
            buffer->strides = buffer->shape + buffer->ndim;
        }

        // Fill in the shape and strides
        const char *metadata = n.get_ndo_meta();
        for (int i = 0; i < buffer->ndim; ++i) {
            switch (dt.get_type_id()) {
                case strided_array_type_id: {
                    const strided_array_dtype *tdt = static_cast<const strided_array_dtype *>(dt.extended());
                    const strided_array_dtype_metadata *md = reinterpret_cast<const strided_array_dtype_metadata *>(metadata);
                    buffer->shape[i] = md->size;
                    buffer->strides[i] = md->stride;
                    metadata += sizeof(strided_array_dtype_metadata);
                    dt = tdt->get_element_dtype();
                    break;
                }
                case fixedarray_type_id: {
                    const fixedarray_dtype *tdt = static_cast<const fixedarray_dtype *>(dt.extended());
                    buffer->shape[i] = tdt->get_fixed_dim_size();
                    buffer->strides[i] = tdt->get_fixed_stride();
                    dt = tdt->get_element_dtype();
                    break;
                }
                default: {
                    stringstream ss;
                    ss << "Cannot get a strided view of dynd dtype " << n.get_dtype() << " for PEP 3118 buffer";
                    throw runtime_error(ss.str());
                }
            }
        }

        // Get the total length of the buffer in bytes
        buffer->len = buffer->itemsize;
        for (int i = 0; i < buffer->ndim; ++i) {
            buffer->len *= buffer->shape[i];
        }

        // Check that any contiguity requirements are satisfied
        if ((flags&PyBUF_C_CONTIGUOUS) || (flags&PyBUF_STRIDES) == 0) {
            if (!strides_are_c_contiguous(buffer->ndim, buffer->itemsize, buffer->shape, buffer->strides)) {
                throw runtime_error("dynd ndobject is not C-contiguous as requested for PEP 3118 buffer");
            }
        } else if (flags&PyBUF_F_CONTIGUOUS) {
            if (!strides_are_f_contiguous(buffer->ndim, buffer->itemsize, buffer->shape, buffer->strides)) {
                throw runtime_error("dynd ndobject is not F-contiguous as requested for PEP 3118 buffer");
            }
        } else if (flags&PyBUF_ANY_CONTIGUOUS) {
            if (!strides_are_c_contiguous(buffer->ndim, buffer->itemsize, buffer->shape, buffer->strides) &&
                    !strides_are_f_contiguous(buffer->ndim, buffer->itemsize, buffer->shape, buffer->strides)) {
                throw runtime_error("dynd ndobject is not C-contiguous nor F-contiguous as requested for PEP 3118 buffer");
            }
        }

        return 0;
    } catch (const std::exception& e) {
        Py_DECREF(ndo);
        buffer->obj = NULL;
        if (buffer->internal != NULL) {
            free(buffer->internal);
            buffer->internal = NULL;
        }
        PyErr_SetString(PyExc_BufferError, e.what());
        return -1;
    }
}

int pydynd::ndobject_releasebuffer_pep3118(PyObject *ndo, Py_buffer *buffer)
{
    try {
        if (buffer->internal != NULL) {
            free(buffer->internal);
            buffer->internal = NULL;
        }
        return 0;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_BufferError, e.what());
        return -1;
    }
}

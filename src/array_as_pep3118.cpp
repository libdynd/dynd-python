//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>

#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/cstruct_type.hpp>
#include <dynd/types/fixedstring_type.hpp>
#include <dynd/types/byteswap_type.hpp>
#include <dynd/types/view_type.hpp>
#include <dynd/shape_tools.hpp>

#include "array_as_pep3118.hpp"
#include "array_functions.hpp"
#include "utility_functions.hpp"

using namespace std;
using namespace dynd;
using namespace pydynd;

static void debug_print_getbuffer_flags(std::ostream& o, int flags)
{
    cout << "Requested buffer flags " << flags << "\n";
    if ((flags&PyBUF_WRITABLE) == PyBUF_WRITABLE) cout << "  PyBUF_WRITABLE\n";
    if ((flags&PyBUF_FORMAT) == PyBUF_FORMAT) cout << "  PyBUF_FORMAT\n";
    if ((flags&PyBUF_ND) == PyBUF_ND) cout << "  PyBUF_ND\n";
    if ((flags&PyBUF_STRIDES) == PyBUF_STRIDES) cout << "  PyBUF_STRIDES\n";
    if ((flags&PyBUF_C_CONTIGUOUS) == PyBUF_C_CONTIGUOUS) cout << "  PyBUF_C_CONTIGUOUS\n";
    if ((flags&PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS) cout << "  PyBUF_F_CONTIGUOUS\n";
    if ((flags&PyBUF_ANY_CONTIGUOUS) == PyBUF_ANY_CONTIGUOUS) cout << "  PyBUF_ANY_CONTIGUOUS\n";
    if ((flags&PyBUF_INDIRECT) == PyBUF_INDIRECT) cout << "  PyBUF_INDIRECT\n";
}

static void debug_print_py_buffer(std::ostream& o, const Py_buffer *buffer, int flags)
{
    cout << "PEP 3118 buffer info:\n";
    cout << "  buf: " << buffer->buf << "\n";
    cout << "  obj: " << (void *)buffer->obj << "\n";
    cout << "  len: " << buffer->len << "\n";
    cout << "  itemsize: " << buffer->itemsize << "\n";
    cout << "  readonly: " << buffer->readonly << "\n";
    cout << "  ndim: " << buffer->ndim << "\n";
    cout << "  format: " << (buffer->format ? buffer->format : "<NULL>") << "\n";
    cout << "  shape: ";
    for (int i = 0; i < buffer->ndim; ++i) cout << buffer->shape[i] << " ";
    cout << "\n";
    cout << "  strides: ";
    for (int i = 0; i < buffer->ndim; ++i) cout << buffer->strides[i] << " ";
    cout << "\n";
    cout << "  internal: " << buffer->internal << endl;
}

static void append_pep3118_format(intptr_t& out_itemsize, const ndt::type& dt, const char *metadata, std::stringstream& o)
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
            switch (static_cast<const fixedstring_type *>(dt.extended())->get_encoding()) {
                case string_encoding_ascii: {
                    intptr_t element_size = dt.get_data_size();
                    o << element_size << "s";
                    out_itemsize = element_size;
                    return;
                }
                // TODO: Couldn't find documentation for UCS-2 character code?
                case string_encoding_utf_32: {
                    intptr_t element_size = dt.get_data_size();
                    o << (element_size/4) << "w";
                    out_itemsize = element_size;
                    return;
                }
                default:
                    break;
            }
            // Pass through to error
            break;
        case fixed_dim_type_id: {
            ndt::type child_dt = dt;
            o << "(";
            do {
                const fixed_dim_type *tdt = static_cast<const fixed_dim_type *>(child_dt.extended());
                size_t dim_size = tdt->get_fixed_dim_size();
                o << dim_size;
                if (child_dt.get_data_size() != tdt->get_element_type().get_data_size() * dim_size) {
                    stringstream ss;
                    ss << "Cannot convert dynd type " << dt << " into a PEP 3118 format because it is not C-order";
                    throw runtime_error(ss.str());
                }
                o << ")";
                child_dt = tdt->get_element_type();
            } while (child_dt.get_type_id() == fixed_dim_type_id && (o << ","));
            append_pep3118_format(out_itemsize, child_dt, metadata, o);
            out_itemsize = dt.get_data_size();
            return;
        }
        case cstruct_type_id: {
            o << "T{";
            const cstruct_type *tdt = static_cast<const cstruct_type *>(dt.extended());
            const ndt::type *field_types = tdt->get_field_types();
            const string *field_names = tdt->get_field_names();
            size_t num_fields = tdt->get_field_count();
            const size_t *offsets = tdt->get_data_offsets(metadata);
            const size_t *metadata_offsets = tdt->get_metadata_offsets();
            size_t format_offset = 0;
            for (size_t i = 0; i != num_fields; ++i) {
                size_t offset = offsets[i];
                // Add padding bytes
                while (offset > format_offset) {
                    o << "x";
                    ++format_offset;
                }
                // The field's type
                append_pep3118_format(out_itemsize, field_types[i], metadata ? (metadata + metadata_offsets[i]) : NULL, o);
                format_offset += out_itemsize;
                // Append the name
                o << ":" << field_names[i] << ":";
            }
            out_itemsize = dt.get_data_size();
            // Add padding bytes to the end
            while ((size_t)out_itemsize > format_offset) {
                o << "x";
                ++format_offset;
            }
            o << "}";
            return;
        }
        case byteswap_type_id: {
            union {
                char s[2];
                uint16_t u;
            } vals;
            vals.u = '>' + ('<' << 8);
            const byteswap_type *bd = static_cast<const byteswap_type *>(dt.extended());
            o << vals.s[0];
            append_pep3118_format(out_itemsize, bd->get_value_type(), metadata, o);
            return;
        }
        case view_type_id: {
            const view_type *vd = static_cast<const view_type *>(dt.extended());
            // If it's a view of bytes, usually to view unaligned data, can ignore it
            // since the buffer format we're creating doesn't use alignment
            if (vd->get_operand_type().get_type_id() == fixedbytes_type_id) {
                append_pep3118_format(out_itemsize, vd->get_value_type(), metadata, o);
                return;
            }
            break;
        }
        default:
            break;
    }
    stringstream ss;
    ss << "Cannot convert dynd type " << dt << " into a PEP 3118 format string";
    throw runtime_error(ss.str());
}

std::string pydynd::make_pep3118_format(intptr_t& out_itemsize, const ndt::type& dt, const char *metadata)
{
    std::stringstream result;
    // Specify native alignment/storage if it's a builtin scalar type
    if (dt.extended() == NULL) {
        result << "@";
    }
    append_pep3118_format(out_itemsize, dt, metadata, result);
    return result.str();
}

static void array_getbuffer_pep3118_bytes(const ndt::type& dt, const char *metadata,
                char *data, Py_buffer *buffer, int flags)
{
    buffer->itemsize = 1;
    if (flags&PyBUF_FORMAT) {
        buffer->format = (char *)"c";
    } else {
        buffer->format = NULL;
    }
    buffer->ndim = 1;
#if PY_VERSION_HEX == 0x02070000
    buffer->internal = NULL;
    buffer->shape = &buffer->smalltable[0];
    buffer->strides = &buffer->smalltable[1];
#else
    buffer->internal = malloc(2*sizeof(intptr_t));
    buffer->shape = reinterpret_cast<Py_ssize_t *>(buffer->internal);
    buffer->strides = buffer->shape + 1;
#endif
    buffer->strides[0] = 1;

    if (dt.get_type_id() == bytes_type_id) {
        // Variable-length bytes type
        char **bytes_data = reinterpret_cast<char **>(data);
        buffer->buf = bytes_data[0];
        buffer->len = (bytes_data[1] - bytes_data[0]);
    } else {
        // Fixed-length bytes type
        buffer->len = dt.get_data_size();
    }
    buffer->shape[0] = buffer->len;
}

int pydynd::array_getbuffer_pep3118(PyObject *ndo, Py_buffer *buffer, int flags)
{
    //debug_print_getbuffer_flags(cout, flags);
    try {
        buffer->shape = NULL;
        buffer->strides = NULL;
        buffer->suboffsets = NULL;
        buffer->format = NULL;
        buffer->obj = ndo;
        buffer->internal = NULL;
        Py_INCREF(ndo);
        if (!WArray_Check(ndo)) {
            throw runtime_error("array_getbuffer_pep3118 called on a non-array");
        }
        nd::array& n = ((WArray *)ndo)->v;
        array_preamble *preamble = n.get_ndo();
        ndt::type dt = n.get_type();

        // Check if a writable buffer is requested
        if ((flags&PyBUF_WRITABLE) && !(n.get_access_flags()&nd::write_access_flag)) {
            throw runtime_error("dynd array is not writable");
        }
        buffer->readonly = ((n.get_access_flags()&nd::write_access_flag) == 0);
        buffer->buf = preamble->m_data_pointer;

        if (dt.get_type_id() == bytes_type_id || dt.get_type_id() == fixedbytes_type_id) {
            array_getbuffer_pep3118_bytes(dt, n.get_ndo_meta(), n.get_ndo()->m_data_pointer, buffer, flags);
            return 0;
        }

        buffer->ndim = (int)dt.get_ndim();
        if (((flags&PyBUF_ND) != PyBUF_ND) && buffer->ndim > 1) {
            stringstream ss;
            ss << "dynd type " << n.get_type() << " is multidimensional, but PEP 3118 request is not ND";
            throw runtime_error(ss.str());
        }

        // Create the format, and allocate the dynamic memory but Py_buffer needs
        char *uniform_metadata = n.get_ndo_meta();
        ndt::type uniform_tp = dt.get_type_at_dimension(&uniform_metadata, buffer->ndim);
        if ((flags&PyBUF_FORMAT) || uniform_tp.get_data_size() == 0) {
            // If the array data type doesn't have a fixed size, make_pep3118 fills buffer->itemsize as a side effect
            string format = make_pep3118_format(buffer->itemsize, uniform_tp, uniform_metadata);
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
            buffer->itemsize = uniform_tp.get_data_size();
            buffer->internal = malloc(2*buffer->ndim*sizeof(intptr_t));
            buffer->shape = reinterpret_cast<Py_ssize_t *>(buffer->internal);
            buffer->strides = buffer->shape + buffer->ndim;
        }

        // Fill in the shape and strides
        const char *metadata = n.get_ndo_meta();
        for (int i = 0; i < buffer->ndim; ++i) {
            switch (dt.get_type_id()) {
                case strided_dim_type_id: {
                    const strided_dim_type *tdt = static_cast<const strided_dim_type *>(dt.extended());
                    const strided_dim_type_metadata *md = reinterpret_cast<const strided_dim_type_metadata *>(metadata);
                    buffer->shape[i] = md->size;
                    buffer->strides[i] = md->stride;
                    metadata += sizeof(strided_dim_type_metadata);
                    dt = tdt->get_element_type();
                    break;
                }
                case fixed_dim_type_id: {
                    const fixed_dim_type *tdt = static_cast<const fixed_dim_type *>(dt.extended());
                    buffer->shape[i] = tdt->get_fixed_dim_size();
                    buffer->strides[i] = tdt->get_fixed_stride();
                    dt = tdt->get_element_type();
                    break;
                }
                default: {
                    stringstream ss;
                    ss << "Cannot get a strided view of dynd type " << n.get_type() << " for PEP 3118 buffer";
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
                throw runtime_error("dynd array is not C-contiguous as requested for PEP 3118 buffer");
            }
        } else if (flags&PyBUF_F_CONTIGUOUS) {
            if (!strides_are_f_contiguous(buffer->ndim, buffer->itemsize, buffer->shape, buffer->strides)) {
                throw runtime_error("dynd array is not F-contiguous as requested for PEP 3118 buffer");
            }
        } else if (flags&PyBUF_ANY_CONTIGUOUS) {
            if (!strides_are_c_contiguous(buffer->ndim, buffer->itemsize, buffer->shape, buffer->strides) &&
                    !strides_are_f_contiguous(buffer->ndim, buffer->itemsize, buffer->shape, buffer->strides)) {
                throw runtime_error("dynd array is not C-contiguous nor F-contiguous as requested for PEP 3118 buffer");
            }
        }

        //debug_print_py_buffer(cout, buffer, flags);

        return 0;
    } catch (const std::exception& e) {
        // Numpy likes to hide these errors and repeatedly try again, so it's useful to see what's happening
        //cout << "ERROR " << e.what() << endl;
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

int pydynd::array_releasebuffer_pep3118(PyObject *ndo, Py_buffer *buffer)
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

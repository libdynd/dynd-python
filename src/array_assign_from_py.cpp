//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>

#include <dynd/typed_data_assign.hpp>
#include <dynd/types/string_type.hpp>
#include <dynd/types/bytes_type.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/fixed_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/types/base_struct_type.hpp>
#include <dynd/types/date_type.hpp>
#include <dynd/types/type_type.hpp>
#include <dynd/memblock/external_memory_block.hpp>
#include <dynd/memblock/pod_memory_block.hpp>
#include <dynd/type_promotion.hpp>
#include <dynd/kernels/assignment_kernels.hpp>

#include "array_from_py.hpp"
#include "array_assign_from_py.hpp"
#include "array_functions.hpp"
#include "type_functions.hpp"
#include "utility_functions.hpp"
#include "numpy_interop.hpp"

using namespace std;
using namespace dynd;
using namespace pydynd;

static size_t get_pyseq_ndim(PyObject *seq, bool& ends_in_dict)
{
    size_t ndim = 0;
    pyobject_ownref obj(seq, true);
    Py_ssize_t seqsize = 0;
    ends_in_dict = false;
    do {
        // Iteratively index the first element until we run out of dimensions
        if (PyDict_Check(obj.get())) {
            ends_in_dict = true;
            seqsize = 0;
        } else if (PyUnicode_Check(obj.get())
#if PY_VERSION_HEX < 0x03000000
                        || PyString_Check(obj.get())
#endif
                        ) {
            // Strings in Python are also sequences and iterable,
            // so need to special case them here
            seqsize = 0;
        } else if (PySequence_Check(obj.get())) {
            Py_ssize_t size = PySequence_Size(obj.get());
            if (size == -1 && PyErr_Occurred()) {
                seqsize = 0;
                PyErr_Clear();
            } else {
                ++ndim;
                seqsize = size;
                if (seqsize > 0) {
                    obj.reset(PySequence_GetItem(obj.get(), 0));
                }
            }
        } else {
            // If we run into an iterator, assume it's exactly one dimension
            PyObject *iter = PyObject_GetIter(obj);
            if (iter == NULL) {
                if (PyErr_ExceptionMatches(PyExc_TypeError)) {
                    // Non-iterators are signalled via the TypeError
                    PyErr_Clear();
                } else {
                    // Propagate any errors
                    throw exception();
                }
            } else {
                Py_DECREF(iter);
                ++ndim;
                break;
            }
            seqsize = 0;
        }
    } while (seqsize > 0);
    return ndim;
}

static void array_assign_from_pyseq(const dynd::ndt::type& dt,
                const char *metadata, char *data, PyObject *seq, size_t seqsize);
static void array_assign_from_pyiter(const dynd::ndt::type& dt,
                const char *metadata, char *data, PyObject *iter, PyObject *obj);
static void array_assign_from_pydict(const dynd::ndt::type& dt,
                const char *metadata, char *data, PyObject *value);

static void array_assign_from_value(const dynd::ndt::type& dt,
                const char *metadata, char *data, PyObject *value)
{
    if (dt.get_ndim() > 0) {
        if (PySequence_Check(value)) {
            Py_ssize_t seqsize = PySequence_Size(value);
            if (seqsize == -1 && PyErr_Occurred()) {
                PyErr_Clear();
            } else {
                array_assign_from_pyseq(dt, metadata, data, value, seqsize);
                return;
            }
        }

        // Try and get an iterator from the object
        PyObject *iter = PyObject_GetIter(value);
        if (iter != NULL) {
            // Since iter was valid, give it a smart pointer as owner
            pyobject_ownref iter_owner(iter);
            array_assign_from_pyiter(dt, metadata, data, iter, value);
            return;
        }

        stringstream ss;
        ss << "error assigning from python object to dynd type " << dt;
        ss << ", expected a sequence as input";
        throw runtime_error(ss.str());
    } else {
        // Do some special case assignments
        if (WArray_Check(value)) {
            const nd::array& v = ((WArray *)value)->v;
            typed_data_assign(dt, metadata, data,
                        v.get_type(), v.get_ndo_meta(), v.get_readonly_originptr());
        } else if (PyBool_Check(value)) {
            dynd_bool v = (value == Py_True);
            typed_data_assign(dt, metadata, data,
                        ndt::make_type<dynd_bool>(), NULL, reinterpret_cast<const char *>(&v));
    #if PY_VERSION_HEX < 0x03000000
        } else if (PyInt_Check(value)) {
            long v = PyInt_AS_LONG(value);
            typed_data_assign(dt, metadata, data,
                        ndt::make_type<long>(), NULL, reinterpret_cast<const char *>(&v));
    #endif // PY_VERSION_HEX < 0x03000000
        } else if (PyLong_Check(value)) {
            PY_LONG_LONG v = PyLong_AsLongLong(value);
            if (v == -1 && PyErr_Occurred()) {
                throw runtime_error("error converting int value");
            }
            typed_data_assign(dt, metadata, data,
                        ndt::make_type<PY_LONG_LONG>(), NULL, reinterpret_cast<const char *>(&v));
        } else if (PyFloat_Check(value)) {
            double v = PyFloat_AS_DOUBLE(value);
            typed_data_assign(dt, metadata, data,
                        ndt::make_type<double>(), NULL, reinterpret_cast<const char *>(&v));
        } else if (PyComplex_Check(value)) {
            complex<double> v(PyComplex_RealAsDouble(value), PyComplex_ImagAsDouble(value));
            typed_data_assign(dt, metadata, data,
                        ndt::make_type<complex<double> >(), NULL, reinterpret_cast<const char *>(&v));
#if PY_VERSION_HEX < 0x03000000
        } else if (PyString_Check(value)) {
            char *pystr_data = NULL;
            intptr_t pystr_len = 0;
            if (PyString_AsStringAndSize(value, &pystr_data, &pystr_len) < 0) {
                throw runtime_error("Error getting string data");
            }

            ndt::type str_dt;
            // Choose between bytes or ascii string based on the destination type
            type_kind_t kind = dt.get_dtype().get_kind();
            if (kind == bytes_kind) {
                str_dt = ndt::make_bytes(1);
            } else { 
                str_dt = ndt::make_string(string_encoding_ascii);
            }
            string_type_data str_d;
            string_type_metadata str_md;
            str_d.begin = pystr_data;
            str_d.end = pystr_data + pystr_len;
            str_md.blockref = NULL;

            typed_data_assign(dt, metadata, data,
                        str_dt, reinterpret_cast<const char *>(&str_md), reinterpret_cast<const char *>(&str_d));
#else
        } else if (PyBytes_Check(value)) {
            char *pybytes_data = NULL;
            intptr_t pybytes_len = 0;
            if (PyBytes_AsStringAndSize(value, &pybytes_data, &pybytes_len) < 0) {
                throw runtime_error("Error getting byte string data");
            }

            ndt::type bytes_dt = ndt::make_bytes(1);
            string_type_data bytes_d;
            string_type_metadata bytes_md;
            bytes_d.begin = pybytes_data;
            bytes_d.end = pybytes_data + pybytes_len;
            bytes_md.blockref = NULL;

            typed_data_assign(dt, metadata, data,
                        bytes_dt, reinterpret_cast<const char *>(&bytes_md), reinterpret_cast<const char *>(&bytes_d));
#endif
        } else if (PyUnicode_Check(value)) {
            // Go through UTF8 (was accessing the cpython uniocde values directly
            // before, but on Python 3.3 OS X it didn't work correctly.)
            pyobject_ownref utf8(PyUnicode_AsUTF8String(value));
            char *s = NULL;
            Py_ssize_t len = 0;
            if (PyBytes_AsStringAndSize(utf8.get(), &s, &len) < 0) {
                throw exception();
            }

            ndt::type str_dt = ndt::make_string(string_encoding_utf_8);
            string_type_data str_d;
            string_type_metadata str_md;
            str_d.begin = s;
            str_d.end = s + len;
            str_md.blockref = NULL;

            typed_data_assign(dt, metadata, data,
                        str_dt, reinterpret_cast<const char *>(&str_md), reinterpret_cast<const char *>(&str_d));
    #if DYND_NUMPY_INTEROP
        } else if (PyArray_Check(value)) {
            const nd::array& v = array_from_numpy_array((PyArrayObject *)value, 0, false);
            typed_data_assign(dt, metadata, data,
                        v.get_type(), v.get_ndo_meta(), v.get_readonly_originptr());
        } else if (PyArray_IsScalar(value, Generic)) {
            const nd::array& v = array_from_numpy_scalar(value, 0);
            typed_data_assign(dt, metadata, data,
                        v.get_type(), v.get_ndo_meta(), v.get_readonly_originptr());
        } else if (PyArray_DescrCheck(value)) {
            const ndt::type& v = make_ndt_type_from_pyobject(value);
            typed_data_assign(dt, metadata, data,
                        ndt::make_type(), NULL, reinterpret_cast<const char *>(&v));
    #endif // DYND_NUMPY_INTEROP
        } else if (WType_Check(value)) {
            const ndt::type& v = ((WType *)value)->v;
            typed_data_assign(dt, metadata, data,
                        ndt::make_type(), NULL, reinterpret_cast<const char *>(&v));
        } else if (PyType_Check(value)) {
            const ndt::type& v = make_ndt_type_from_pyobject(value);
            typed_data_assign(dt, metadata, data,
                        ndt::make_type(), NULL, reinterpret_cast<const char *>(&v));
        } else if (PyDict_Check(value)) {
            array_assign_from_pydict(dt, metadata, data, value);
        } else {
            // Check if the value is a sequence
            if (PySequence_Check(value)) {
                Py_ssize_t seqsize = PySequence_Size(value);
                if (seqsize == -1 && PyErr_Occurred()) {
                    PyErr_Clear();
                } else {
                    array_assign_from_pyseq(dt, metadata, data, value, seqsize);
                    return;
                }
            }

            // Fall back strategy, where we convert to nd::array, then assign
            nd::array v = array_from_py(value, 0, false);
            typed_data_assign(dt, metadata, data,
                            v.get_type(), v.get_ndo_meta(), v.get_readonly_originptr());
        }
    }
}

static void array_assign_strided_from_pyseq(const dynd::ndt::type& element_dt,
                const char *element_metadata, char *dst_data, intptr_t dst_stride, size_t dst_size,
                PyObject *seq, size_t seqsize)
{
    // Raise a broadcast error if it doesn't work
    if (seqsize != 1 && dst_size != seqsize) {
        throw broadcast_error(element_dt, element_metadata, "nested python sequence object");
    }
    if (seqsize == 1 && dst_size > 1) {
        // If there's one input value and many outputs,
        // convert it from Python once and copy to the rest
        pyobject_ownref item(PySequence_GetItem(seq, 0));
        array_assign_from_value(element_dt, element_metadata, dst_data, item.get());
        // Get a strided kernel, and use it to assign from the first element to the rest of them
        assignment_ckernel_builder k;
        make_assignment_kernel(&k, 0, element_dt, element_metadata,
                        element_dt, element_metadata, kernel_request_strided,
                        assign_error_default, &eval::default_eval_context);
        ckernel_prefix *kdp = k.get();
        kdp->get_function<unary_strided_operation_t>()(
                        dst_data + dst_stride, dst_stride,
                        dst_data, 0, dst_size - 1, kdp);
    } else {
        // Loop through the dst array and assign values
        for (size_t i = 0; i != dst_size; ++i) {
            pyobject_ownref item(PySequence_GetItem(seq, i));
            array_assign_from_value(element_dt, element_metadata,
                            dst_data + i * dst_stride, item.get());
        }
    }
}

/**
 * Assigns from a Python iterator to a strided array, obeying
 * the standard broadcasting rules. This is trickier than from an array
 * with known size, because we must detect the 1 -> N broadcasting case
 * by trying to get the second element and broadcasting on StopIteration
 * or copy elementwise in the other case.
 */
static void array_assign_strided_from_pyiter(const dynd::ndt::type& element_dt,
                const char *element_metadata, char *dst_data, intptr_t dst_stride, size_t dst_size,
                PyObject *iter)
{
    // If the destination size is zero, the iteration should stop immediately
    if (dst_size == 0) {
        PyObject *item = PyIter_Next(iter);
        if (item != NULL) {
            Py_DECREF(item);
            // Too many fields provided
            stringstream ss;
            ss << "python iterator input for size " << dst_size << " array of dynd type " << element_dt;
            ss << " had too many values.";
            throw broadcast_error(ss.str());
        } else if (PyErr_Occurred()) {
            // Propagate any exception
            throw exception();
        }
        return;
    }

    // Get the first element of the sequence
    PyObject *item = PyIter_Next(iter);
    if (item == NULL) {
        if (PyErr_Occurred()) {
            // Propagate any exception
            throw exception();
        } else {
            // Not enough values provided
            stringstream ss;
            ss << "python iterator input for size " << dst_size << " array of dynd type " << element_dt;
            ss << " had too few values.";
            throw broadcast_error(ss.str());
        }
    }
    // Assign ownership of the item and assign it to 
    pyobject_ownref item_owner(item);
    array_assign_from_value(element_dt, element_metadata, dst_data, item);

    // Get the second item, to determine whether to broadcast or not
    item = PyIter_Next(iter);
    if (item == NULL) {
        if (PyErr_Occurred()) {
            // Propagate any exception
            throw exception();
        }
        if (dst_size > 1) {
            // Get a strided kernel, and use it to assign from the first element to the rest of them
            assignment_ckernel_builder k;
            make_assignment_kernel(&k, 0, element_dt, element_metadata,
                            element_dt, element_metadata, kernel_request_strided,
                            assign_error_default, &eval::default_eval_context);
            ckernel_prefix *kdp = k.get();
            kdp->get_function<unary_strided_operation_t>()(
                            dst_data + dst_stride, dst_stride,
                            dst_data, 0, dst_size - 1, kdp);
        }
    } else {
        // Copy the second item, and the rest
        for (size_t i = 1; i != dst_size; ++i) {
            if (item == NULL) {
                if (PyErr_Occurred()) {
                    // Propagate any exception
                    throw exception();
                } else {
                    // Not enough values provided
                    stringstream ss;
                    ss << "python iterator input for size " << dst_size << " array of dynd type " << element_dt;
                    ss << " had too few values.";
                    throw broadcast_error(ss.str());
                }
            }
            // Assign ownership of 'item' so it will be freed as appropriate
            item_owner.reset(item);
            array_assign_from_value(element_dt, element_metadata,
                            dst_data + i * dst_stride, item);
            // Get the next item
            item = PyIter_Next(iter);
        }

        // Make sure that the last PyIter_Next called returned NULL and didn't set an error
        if (item != NULL) {
            Py_DECREF(item);
            // Too many values provided
            stringstream ss;
            ss << "python iterator input for size " << dst_size << " array of dynd type " << element_dt;
            ss << " had too many values.";
            throw broadcast_error(ss.str());
        } else if (PyErr_Occurred()) {
            // Propagate any exception
            throw exception();
        }
    }
}

static void array_assign_from_pydict(const dynd::ndt::type& dt,
                const char *metadata, char *data, PyObject *value)
{
    if (dt.get_kind() == struct_kind) {
        const base_struct_type *fsd = static_cast<const base_struct_type *>(dt.extended());
        size_t field_count = fsd->get_field_count();
        const string *field_names = fsd->get_field_names();
        const ndt::type *field_types = fsd->get_field_types();
        const size_t *data_offsets = fsd->get_data_offsets(metadata);
        const size_t *metadata_offsets = fsd->get_metadata_offsets();

        // Keep track of which fields we've seen
        shortvector<bool> populated_fields(field_count);
        memset(populated_fields.get(), 0, sizeof(bool) * field_count);

        PyObject *dict_key = NULL, *dict_value = NULL;
        Py_ssize_t dict_pos = 0;

        while (PyDict_Next(value, &dict_pos, &dict_key, &dict_value)) {
            string name = pystring_as_string(dict_key);
            intptr_t i = fsd->get_field_index(name);
            // TODO: Add an error policy of whether to throw an error
            //       or not. For now, just raise an error
            if (i != -1) {
                array_assign_from_value(field_types[i], metadata + metadata_offsets[i],
                                data + data_offsets[i], dict_value);
                populated_fields[i] = true;
            } else {
                stringstream ss;
                ss << "Input python dict has key ";
                print_escaped_utf8_string(ss, name);
                ss << ", but no such field is in destination dynd type " << dt;
                throw runtime_error(ss.str());
            }
        }

        for (size_t i = 0; i < field_count; ++i) {
            if (!populated_fields[i]) {
                stringstream ss;
                ss << "python dict does not contain the field ";
                print_escaped_utf8_string(ss, field_names[i]);
                ss << " as required by the data type " << dt;
                throw runtime_error(ss.str());
            }
        }
    } else {
        // TODO support assigning to date_type as well
        stringstream ss;
        ss << "Cannot assign python dict to dynd type " << dt;
        throw runtime_error(ss.str());
    }
}

static void array_assign_from_pyseq(const dynd::ndt::type& dt,
                const char *metadata, char *data, PyObject *seq, size_t seqsize)
{
    switch (dt.get_type_id()) {
        case fixed_dim_type_id: {
            const fixed_dim_type *fdd = static_cast<const fixed_dim_type *>(dt.extended());
            array_assign_strided_from_pyseq(fdd->get_element_type(), metadata,
                            data, fdd->get_fixed_stride(), fdd->get_fixed_dim_size(), seq, seqsize);
            break;
        }
        case strided_dim_type_id: {
            const ndt::type& element_dt = static_cast<const strided_dim_type *>(dt.extended())->get_element_type();
            const strided_dim_type_metadata *md = reinterpret_cast<const strided_dim_type_metadata *>(metadata);
            array_assign_strided_from_pyseq(element_dt, metadata + sizeof(strided_dim_type_metadata),
                            data, md->stride, md->size, seq, seqsize);
            break;
        }
        case var_dim_type_id: {
            const ndt::type& element_dt = static_cast<const var_dim_type *>(dt.extended())->get_element_type();
            const var_dim_type_metadata *md = reinterpret_cast<const var_dim_type_metadata *>(metadata);
            var_dim_type_data *d = reinterpret_cast<var_dim_type_data *>(data);
            if (d->begin == NULL) {
                // Need to allocate the destination data
                if (md->offset != 0) {
                    throw runtime_error("Cannot assign to an uninitialized dynd var_dim which has a non-zero offset");
                }
                intptr_t dst_stride = md->stride;
                // Have to allocate the output
                memory_block_data *memblock = md->blockref;
                if (memblock->m_type == objectarray_memory_block_type) {
                    memory_block_objectarray_allocator_api *allocator =
                                    get_memory_block_objectarray_allocator_api(memblock);

                    // Allocate the output array data
                    d->begin = allocator->allocate(memblock, seqsize);
                } else {
                    memory_block_pod_allocator_api *allocator =
                                    get_memory_block_pod_allocator_api(memblock);

                    // Allocate the output array data
                    char *dst_end = NULL;
                    allocator->allocate(memblock, seqsize * dst_stride,
                                element_dt.get_data_alignment(), &d->begin, &dst_end);
                }
                d->size = seqsize;
            }
            array_assign_strided_from_pyseq(element_dt, metadata + sizeof(var_dim_type_metadata),
                            d->begin + md->offset, md->stride, d->size, seq, seqsize);
            break;
        }
        case struct_type_id:
        case cstruct_type_id: {
            const base_struct_type *fsd = static_cast<const base_struct_type *>(dt.extended());
            size_t field_count = fsd->get_field_count();
            const ndt::type *field_types = fsd->get_field_types();
            const size_t *data_offsets = fsd->get_data_offsets(metadata);
            const size_t *metadata_offsets = fsd->get_metadata_offsets();

            if (seqsize != field_count) {
                stringstream ss;
                ss << "Cannot assign sequence of size " << seqsize;
                ss << " to dynd type " << dt;
                ss << " because it requires " << field_count << " values";
                throw runtime_error(ss.str());
            }
            for (size_t i = 0; i != seqsize; ++i) {
                pyobject_ownref item(PySequence_GetItem(seq, i));
                array_assign_from_value(field_types[i], metadata + metadata_offsets[i],
                                data + data_offsets[i], item.get());
            }
            break;
        }
        default: {
            stringstream ss;
            ss << "Assigning from nested python sequence object to dynd type " << dt;
            ss << " is not yet supported";
            throw runtime_error(ss.str());
        }   
    }
}

static void array_assign_from_pyiter(const dynd::ndt::type& dt,
                const char *metadata, char *data, PyObject *iter, PyObject *obj)
{
    switch (dt.get_type_id()) {
        case fixed_dim_type_id: {
            const fixed_dim_type *fdd = static_cast<const fixed_dim_type *>(dt.extended());
            array_assign_strided_from_pyiter(fdd->get_element_type(), metadata,
                            data, fdd->get_fixed_stride(), fdd->get_fixed_dim_size(), iter);
            break;
        }
        case strided_dim_type_id: {
            const ndt::type& element_dt = static_cast<const strided_dim_type *>(dt.extended())->get_element_type();
            const strided_dim_type_metadata *md = reinterpret_cast<const strided_dim_type_metadata *>(metadata);
            array_assign_strided_from_pyiter(element_dt, metadata + sizeof(strided_dim_type_metadata),
                            data, md->stride, md->size, iter);
            break;
        }
        case var_dim_type_id: {
            const ndt::type& element_dt = static_cast<const var_dim_type *>(dt.extended())->get_element_type();
            const var_dim_type_metadata *md = reinterpret_cast<const var_dim_type_metadata *>(metadata);
            var_dim_type_data *d = reinterpret_cast<var_dim_type_data *>(data);
            // First check if the var_dim element is already assigned,
            // in which case we do a strided assignment
            if (d->begin != NULL) {
                array_assign_strided_from_pyiter(element_dt, metadata + sizeof(var_dim_type_metadata),
                                d->begin + md->offset, md->stride, d->size, iter);
                break;
            }

            // Get the size hint
            intptr_t allocsize = _PyObject_LengthHint(obj, -2);
            if (allocsize == -1) {
                // -1 indicates an exception was raised
                throw exception();
            }

            if (allocsize < 0) {
                // Default to 32 size at start
                allocsize = 32;
            }

            // Need to allocate the destination data
            if (md->offset != 0) {
                throw runtime_error("Cannot assign to an uninitialized dynd var_dim which has a non-zero offset");
            }
            intptr_t filledsize = 0;
            intptr_t dst_stride = md->stride;
            // Have to allocate the output
            memory_block_data *memblock = md->blockref;
            if (memblock->m_type == objectarray_memory_block_type) {
                memory_block_objectarray_allocator_api *allocator =
                                get_memory_block_objectarray_allocator_api(memblock);

                // Allocate the output array data
                d->begin = allocator->allocate(memblock, allocsize);
                d->size = 0;

                PyObject *item;
                while ((item = PyIter_Next(iter))) {
                    pyobject_ownref item_ref(item);

                    if (allocsize == filledsize) {
                        allocsize = 3 * allocsize / 2;
                        d->begin = allocator->resize(memblock, d->begin, allocsize);
                    }

                    array_assign_from_value(element_dt, metadata + sizeof(var_dim_type_metadata),
                                    d->begin + filledsize * md->stride, item);
                    ++filledsize;
                }

                if (PyErr_Occurred()) {
                    allocator->resize(memblock, d->begin, 0);
                    d->begin = NULL;
                    throw exception();
                } else {
                    // If we didn't use all the space, shrink to fit
                    if (filledsize < allocsize) {
                        d->begin = allocator->resize(memblock, d->begin, filledsize);
                    }
                    d->size = filledsize;
                }
            } else {
                memory_block_pod_allocator_api *allocator =
                                get_memory_block_pod_allocator_api(memblock);

                // Allocate the output array data
                char *dst_end = NULL;
                allocator->allocate(memblock, allocsize * dst_stride,
                            element_dt.get_data_alignment(), &d->begin, &dst_end);
                d->size = 0;

                PyObject *item;
                while ((item = PyIter_Next(iter))) {
                    pyobject_ownref item_ref(item);

                    if (allocsize == filledsize) {
                        allocsize = 3 * allocsize / 2;
                        allocator->resize(memblock, allocsize * dst_stride, &d->begin, &dst_end);
                    }

                    array_assign_from_value(element_dt, metadata + sizeof(var_dim_type_metadata),
                                    d->begin + filledsize * md->stride, item);
                    ++filledsize;
                }

                if (PyErr_Occurred()) {
                    allocator->resize(memblock, 0, &d->begin, &dst_end);
                    d->begin = NULL;
                    throw exception();
                } else {
                    // If we didn't use all the space, shrink to fit
                    if (filledsize < allocsize) {
                        allocator->resize(memblock, filledsize * dst_stride, &d->begin, &dst_end);
                    }
                    d->size = filledsize;
                }
            }
            break;
        }
        case struct_type_id:
        case cstruct_type_id: {
            const base_struct_type *fsd = static_cast<const base_struct_type *>(dt.extended());
            size_t field_count = fsd->get_field_count();
            const ndt::type *field_types = fsd->get_field_types();
            const size_t *data_offsets = fsd->get_data_offsets(metadata);
            const size_t *metadata_offsets = fsd->get_metadata_offsets();

            for (size_t i = 0; i != field_count; ++i) {
                PyObject *item = PyIter_Next(iter);
                if (item == NULL) {
                    if (PyErr_Occurred()) {
                        // Propagate any exception
                        throw exception();
                    } else {
                        // Not enough fields provided
                        stringstream ss;
                        ss << "python iterator input for dynd type " << dt;
                        ss << " did not have enough fields.";
                        throw broadcast_error(ss.str());
                    }
                }
                pyobject_ownref item_owner(item);
                array_assign_from_value(field_types[i], metadata + metadata_offsets[i],
                                data + data_offsets[i], item);
            }

            // Make sure there are no more elements provided by the iterator
            PyObject *item = PyIter_Next(iter);
            if (item != NULL) {
                Py_DECREF(item);
                // Too many fields provided
                stringstream ss;
                ss << "python iterator input for dynd type " << dt;
                ss << " had too many fields.";
                throw broadcast_error(ss.str());
            } else if (PyErr_Occurred()) {
                // Propagate any exception
                throw exception();
            }
            break;
        }
        default: {
            stringstream ss;
            ss << "Assigning from nested python sequence object to dynd type " << dt;
            ss << " is not yet supported";
            throw runtime_error(ss.str());
        }   
    }
}

void pydynd::array_broadcast_assign_from_py(const dynd::ndt::type& dt,
                const char *metadata, char *data, PyObject *value)
{
    // Special-case assigning from known array types
    if (WArray_Check(value)) {
        const nd::array& n = ((WArray *)value)->v;
        typed_data_assign(dt, metadata, data,
                        n.get_type(), n.get_ndo_meta(), n.get_readonly_originptr());
        return;
#if DYND_NUMPY_INTEROP
    } else if (PyArray_Check(value)) {
        const nd::array& v = array_from_numpy_array((PyArrayObject *)value, 0, false);
        typed_data_assign(dt, metadata, data,
                    v.get_type(), v.get_ndo_meta(), v.get_readonly_originptr());
        return;
#endif // DYND_NUMPY_INTEROP
    }

    size_t dst_ndim = dt.get_ndim();
    bool ends_in_dict = false;
    if (dst_ndim == 0) {
        array_assign_from_value(dt, metadata, data, value);
    } else {
        size_t seq_ndim = get_pyseq_ndim(value, ends_in_dict);
        // Special handling when the destination is a struct,
        // and there was no dict at the end of the seq chain.
        // Increase the dst_ndim to count the first field
        // of any structs as uniform, to match up with how
        // get_pyseq_ndim works.
        ndt::type udt = dt.get_dtype().value_type();
        size_t original_dst_ndim = dst_ndim;
        if (!ends_in_dict && udt.get_kind() == struct_kind) {
            while (true) {
                if (udt.get_ndim() > 0) {
                    dst_ndim += udt.get_ndim();
                    udt = udt.get_dtype();
                } else if (udt.get_kind() == struct_kind) {
                    ++dst_ndim;
                    udt = static_cast<const base_struct_type *>(udt.extended())->get_field_types()[0];
                } else {
                    break;
                }
            }
        }

        if (dst_ndim > seq_ndim || dt.is_expression()) {
            // Make a temporary value with just the trailing dimensions, then
            // assign to the output
            dimvector shape(original_dst_ndim);
            dt.extended()->get_shape(original_dst_ndim, 0, shape.get(), metadata);
            ndt::type partial_dt = dt.get_type_at_dimension(NULL, dst_ndim - seq_ndim).get_canonical_type();
            nd::array tmp(make_array_memory_block(partial_dt, original_dst_ndim - (dst_ndim - seq_ndim),
                            shape.get() + (dst_ndim - seq_ndim)));
            array_assign_from_value(tmp.get_type(), tmp.get_ndo_meta(), tmp.get_readwrite_originptr(),
                            value);
            typed_data_assign(dt, metadata, data, tmp.get_type(), tmp.get_ndo_meta(), tmp.get_readonly_originptr());
        } else {
            array_assign_from_value(dt, metadata, data, value);
        }
    }
}

void pydynd::array_nodim_broadcast_assign_from_py(const dynd::ndt::type& dt, const char *metadata, char *data, PyObject *value)
{
    array_assign_from_value(dt, metadata, data, value);
}

void pydynd::array_broadcast_assign_from_py(const dynd::nd::array& n, PyObject *value)
{
    array_broadcast_assign_from_py(n.get_type(), n.get_ndo_meta(), n.get_readwrite_originptr(), value);
}

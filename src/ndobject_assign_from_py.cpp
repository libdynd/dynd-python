//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>
#include <datetime.h>

#include <dynd/dtype_assign.hpp>
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
#include <dynd/kernels/assignment_kernels.hpp>

#include "ndobject_from_py.hpp"
#include "ndobject_assign_from_py.hpp"
#include "ndobject_functions.hpp"
#include "dtype_functions.hpp"
#include "utility_functions.hpp"
#include "numpy_interop.hpp"

using namespace std;
using namespace dynd;
using namespace pydynd;

static size_t get_pyseq_ndim(PyObject *seq)
{
    size_t ndim = 0;
    pyobject_ownref obj(seq, true);
    Py_ssize_t seqsize = 0;
    do {
        // Iteratively index the first element until we run out of dimensions
        if (PySequence_Check(obj.get())) {
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
            seqsize = 0;
        }
    } while (seqsize > 0);
    return ndim;
}

static void ndobject_assign_from_pyseq(const dynd::dtype& dt,
                const char *metadata, char *data, PyObject *seq, size_t seqsize);

static void ndobject_assign_from_value(const dynd::dtype& dt,
                const char *metadata, char *data, PyObject *value)
{
    if (dt.get_undim() > 0) {
        if (PySequence_Check(value)) {
            Py_ssize_t seqsize = PySequence_Size(value);
            if (seqsize == -1 && PyErr_Occurred()) {
                PyErr_Clear();
            } else {
                ndobject_assign_from_pyseq(dt, metadata, data, value, seqsize);
                return;
            }
        }

        stringstream ss;
        ss << "error assigning from python object to dynd type " << dt;
        ss << ", expected a sequence as input";
        throw runtime_error(ss.str());
    } else {
        // Do some special case assignments
        if (WNDObject_Check(value)) {
            const ndobject& v = ((WNDObject *)value)->v;
            dtype_assign(dt, metadata, data,
                        v.get_dtype(), v.get_ndo_meta(), v.get_readonly_originptr());
        } else if (PyBool_Check(value)) {
            dynd_bool v = (value == Py_True);
            dtype_assign(dt, metadata, data,
                        make_dtype<dynd_bool>(), NULL, reinterpret_cast<const char *>(&v));
    #if PY_VERSION_HEX < 0x03000000
        } else if (PyInt_Check(value)) {
            long v = PyInt_AS_LONG(value);
            dtype_assign(dt, metadata, data,
                        make_dtype<long>(), NULL, reinterpret_cast<const char *>(&v));
    #endif // PY_VERSION_HEX < 0x03000000
        } else if (PyLong_Check(value)) {
            PY_LONG_LONG v = PyLong_AsLongLong(value);
            if (v == -1 && PyErr_Occurred()) {
                throw runtime_error("error converting int value");
            }
            dtype_assign(dt, metadata, data,
                        make_dtype<PY_LONG_LONG>(), NULL, reinterpret_cast<const char *>(&v));
        } else if (PyFloat_Check(value)) {
            double v = PyFloat_AS_DOUBLE(value);
            dtype_assign(dt, metadata, data,
                        make_dtype<double>(), NULL, reinterpret_cast<const char *>(&v));
        } else if (PyComplex_Check(value)) {
            complex<double> v(PyComplex_RealAsDouble(value), PyComplex_ImagAsDouble(value));
            dtype_assign(dt, metadata, data,
                        make_dtype<complex<double> >(), NULL, reinterpret_cast<const char *>(&v));
        } else if (PyString_Check(value)) { // TODO: On Python 3, PyBytes should become a dnd bytes array
            char *pystr_data = NULL;
            intptr_t pystr_len = 0;
            if (PyString_AsStringAndSize(value, &pystr_data, &pystr_len) < 0) {
                throw runtime_error("Error getting string data");
            }

            dtype str_dt = make_string_dtype(string_encoding_ascii);
            string_dtype_data str_d;
            string_dtype_metadata str_md;
            str_d.begin = pystr_data;
            str_d.end = pystr_data + pystr_len;
            str_md.blockref = NULL;

            dtype_assign(dt, metadata, data,
                        str_dt, reinterpret_cast<const char *>(&str_md), reinterpret_cast<const char *>(&str_d));
        } else if (PyUnicode_Check(value)) { // TODO: On Python 3, PyBytes should become a dnd bytes array
    #if Py_UNICODE_SIZE == 2
            dtype str_dt = make_string_dtype(string_encoding_ucs_2);
    #else
            dtype str_dt = make_string_dtype(string_encoding_utf_32);
    #endif
            string_dtype_data str_d;
            string_dtype_metadata str_md;
            str_d.begin = reinterpret_cast<char *>(PyUnicode_AsUnicode(value));
            str_d.end = str_d.begin + Py_UNICODE_SIZE * PyUnicode_GetSize(value);
            str_md.blockref = NULL;

            dtype_assign(dt, metadata, data,
                        str_dt, reinterpret_cast<const char *>(&str_md), reinterpret_cast<const char *>(&str_d));
    #if DYND_NUMPY_INTEROP
        } else if (PyArray_Check(value)) {
            const ndobject& v = ndobject_from_numpy_array((PyArrayObject *)value);
            dtype_assign(dt, metadata, data,
                        v.get_dtype(), v.get_ndo_meta(), v.get_readonly_originptr());
        } else if (PyArray_IsScalar(value, Generic)) {
            const ndobject& v = ndobject_from_numpy_scalar(value);
            dtype_assign(dt, metadata, data,
                        v.get_dtype(), v.get_ndo_meta(), v.get_readonly_originptr());
        } else if (PyArray_DescrCheck(value)) {
            const dtype& v = make_dtype_from_pyobject(value);
            dtype_assign(dt, metadata, data,
                        make_dtype_dtype(), NULL, reinterpret_cast<const char *>(&v));
    #endif // DYND_NUMPY_INTEROP
        } else if (WDType_Check(value)) {
            const dtype& v = ((WDType *)value)->v;
            dtype_assign(dt, metadata, data,
                        make_dtype_dtype(), NULL, reinterpret_cast<const char *>(&v));
        } else if (PyType_Check(value)) {
            const dtype& v = make_dtype_from_pyobject(value);
            dtype_assign(dt, metadata, data,
                        make_dtype_dtype(), NULL, reinterpret_cast<const char *>(&v));
        } else {
            // Fall back strategy, where we convert to ndobject, then assign
            ndobject v = ndobject_from_py(value);
            dtype_assign(dt, metadata, data,
                            v.get_dtype(), v.get_ndo_meta(), v.get_readonly_originptr());
        }
    }
}

static void ndobject_assign_strided_from_pyseq(const dynd::dtype& element_dt,
                const char *element_metadata, char *dst_data, intptr_t dst_stride, size_t dst_size,
                PyObject *seq, size_t seqsize)
{
    // Raise a broadcast error if it doesn't work
    if (seqsize != 1 && dst_size != (intptr_t)dst_size) {
        throw broadcast_error(element_dt, element_metadata, "nested python sequence object");
    }
    if (seqsize == 1 && dst_size > 1) {
        // If there's one input value and many outputs,
        // convert it from Python once and copy to the rest
        pyobject_ownref item(PySequence_GetItem(seq, 0));
        ndobject_assign_from_value(element_dt, element_metadata, dst_data, item.get());
        // Get a strided kernel, and use it to assign from the first element to the rest of them
        assignment_kernel k;
        make_assignment_kernel(&k, 0, element_dt, element_metadata,
                        element_dt, element_metadata, kernel_request_strided,
                        assign_error_default, &eval::default_eval_context);
        kernel_data_prefix *kdp = k.get();
        kdp->get_function<unary_strided_operation_t>()(
                        dst_data + dst_stride, dst_stride,
                        dst_data, 0, dst_size - 1, kdp);
    } else {
        // Loop through the dst array and assign values
        for (intptr_t i = 0; i != dst_size; ++i) {
            pyobject_ownref item(PySequence_GetItem(seq, i));
            ndobject_assign_from_value(element_dt, element_metadata,
                            dst_data + i * dst_stride, item.get());
        }
    }
}

static void ndobject_assign_from_pyseq(const dynd::dtype& dt,
                const char *metadata, char *data, PyObject *seq, size_t seqsize)
{
    switch (dt.get_type_id()) {
        case fixed_dim_type_id: {
            const fixed_dim_dtype *fdd = static_cast<const fixed_dim_dtype *>(dt.extended());
            ndobject_assign_strided_from_pyseq(fdd->get_element_dtype(), metadata,
                            data, fdd->get_fixed_stride(), fdd->get_fixed_dim_size(), seq, seqsize);
            break;
        }
        case strided_dim_type_id: {
            const dtype& element_dt = static_cast<const strided_dim_dtype *>(dt.extended())->get_element_dtype();
            const strided_dim_dtype_metadata *md = reinterpret_cast<const strided_dim_dtype_metadata *>(metadata);
            ndobject_assign_strided_from_pyseq(element_dt, metadata + sizeof(strided_dim_dtype_metadata),
                            data, md->stride, md->size, seq, seqsize);
            break;
        }
        case var_dim_type_id: {
            const dtype& element_dt = static_cast<const var_dim_dtype *>(dt.extended())->get_element_dtype();
            const var_dim_dtype_metadata *md = reinterpret_cast<const var_dim_dtype_metadata *>(metadata);
            var_dim_dtype_data *d = reinterpret_cast<var_dim_dtype_data *>(data);
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
                                element_dt.get_alignment(), &d->begin, &dst_end);
                }
                d->size = seqsize;
            }
            ndobject_assign_strided_from_pyseq(element_dt, metadata + sizeof(var_dim_dtype_metadata),
                            d->begin + md->offset, md->stride, d->size, seq, seqsize);
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

void pydynd::ndobject_broadcast_assign_from_py(const dynd::dtype& dt,
                const char *metadata, char *data, PyObject *value)
{
    size_t dst_undim = dt.get_undim();
    if (dst_undim == 0) {
        ndobject_assign_from_value(dt, metadata, data, value);
    } else {
        size_t seq_undim = get_pyseq_ndim(value);
        if (dst_undim < seq_undim) {
            throw broadcast_error(dt, metadata, "nested python sequence object");
        }  else if (dst_undim > seq_undim || dt.is_expression()) {
            // Make a temporary value with just the trailing dimensions, then
            // assign to the output
            dimvector shape(dst_undim);
            dt.extended()->get_shape(0, shape.get(), metadata);
            dtype partial_dt = dt.get_dtype_at_dimension(NULL, dst_undim - seq_undim).get_canonical_dtype();
            ndobject tmp(make_ndobject_memory_block(partial_dt, seq_undim, shape.get() + (dst_undim - seq_undim)));
            ndobject_assign_from_value(partial_dt, tmp.get_ndo_meta(), tmp.get_readwrite_originptr(),
                            value);
            dtype_assign(dt, metadata, data, partial_dt, tmp.get_ndo_meta(), tmp.get_readonly_originptr());
        } else {
            ndobject_assign_from_value(dt, metadata, data, value);
        }
    }
}


void pydynd::ndobject_broadcast_assign_from_py(const dynd::ndobject& n, PyObject *value)
{
    ndobject_broadcast_assign_from_py(n.get_dtype(), n.get_ndo_meta(), n.get_readwrite_originptr(), value);
}
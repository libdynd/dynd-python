//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>

#include <dynd/dtypes/expr_dtype.hpp>
#include <dynd/dtypes/strided_dim_dtype.hpp>
#include <dynd/dtypes/var_dim_dtype.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>
#include <dynd/kernels/elwise_expr_kernels.hpp>

#include "utility_functions.hpp"
#include "elwise_map.hpp"
#include "ndobject_functions.hpp"
#include "dtype_functions.hpp"

using namespace std;
using namespace dynd;
using namespace pydynd;

namespace {
    struct pyobject_expr_kernel_extra {
        typedef pyobject_expr_kernel_extra extra_type;

        kernel_data_prefix base;
        size_t src_count;
        PyObject *callable;
        // After this are 1 + src_count shell WNDObjects,
        // whose guts are replaced for each call of
        // the kernel

        static void single(char *dst, const char * const *src,
                        kernel_data_prefix *extra)
        {
            PyGILState_STATE gstate;
            gstate = PyGILState_Ensure();

            extra_type *e = reinterpret_cast<extra_type *>(extra);
            WNDObject **ndo = reinterpret_cast<WNDObject **>(e + 1);
            size_t src_count = e->src_count;
            // Modify the temporary ndobjects to point at the data.
            // The constructor sets the metadata to be a size-1 array,
            // so no need to modify any of it.
            ndo[0]->v.get_ndo()->m_data_pointer = dst;
            for (size_t i = 0; i != src_count; ++i) {
                ndo[i+1]->v.get_ndo()->m_data_pointer = const_cast<char *>(src[i]);
            }
            // Put all the ndobjects in a tuple
            pyobject_ownref args(PyTuple_New(src_count + 1));
            for (size_t i = 0; i != src_count + 1; ++i) {
                PyTuple_SET_ITEM(args.get(), i, (PyObject *)ndo[i]);
            }
            // Call the function
            pyobject_ownref res(PyObject_Call(e->callable, args.get(), NULL));
            args.clear();
            // Verify that nothing was returned
            if (res.get() != Py_None) {
                throw runtime_error("Python callable for elwise_map must not return a value, got an object");
            }
            // Verify that no reference to a temporary ndobject was kept
            for (size_t i = 0; i != src_count + 1; ++i) {
                if (ndo[i]->ob_refcnt != 1 || ndo[i]->v.get_ndo()->m_memblockdata.m_use_count != 1) {
                    throw runtime_error("Python callable for elwise_map must not keep a reference "
                                    "to its arguments");
                }
            }

            PyGILState_Release(gstate);
        }

        static void strided(char *dst, intptr_t dst_stride,
                    const char * const *src, const intptr_t *src_stride,
                    size_t count, kernel_data_prefix *extra)
        {
            PyGILState_STATE gstate;
            gstate = PyGILState_Ensure();

            extra_type *e = reinterpret_cast<extra_type *>(extra);
            WNDObject **ndo = reinterpret_cast<WNDObject **>(e + 1);
            size_t src_count = e->src_count;
            strided_dim_dtype_metadata *md;
            // Modify the temporary ndobjects to point at the data.
            // The constructor sets the metadata to be a size-1 array,
            // so no need to modify any of it.
            ndo[0]->v.get_ndo()->m_data_pointer = dst;
            md = reinterpret_cast<strided_dim_dtype_metadata *>(ndo[0]->v.get_ndo_meta());
            md->size = count;
            md->stride = dst_stride;
            for (size_t i = 0; i != src_count; ++i) {
                ndo[i+1]->v.get_ndo()->m_data_pointer = const_cast<char *>(src[i]);
                md = reinterpret_cast<strided_dim_dtype_metadata *>(ndo[i+1]->v.get_ndo_meta());
                md->size = count;
                md->stride = src_stride[i];
            }
            // Put all the ndobjects in a tuple
            pyobject_ownref args(PyTuple_New(src_count + 1));
            for (size_t i = 0; i != src_count + 1; ++i) {
                Py_INCREF(ndo[i]);
                PyTuple_SET_ITEM(args.get(), i, (PyObject *)ndo[i]);
            }
            // Call the function
            pyobject_ownref res(PyObject_Call(e->callable, args.get(), NULL));
            args.clear();
            // Verify that nothing was returned
            if (res.get() != Py_None) {
                throw runtime_error("Python callable for elwise_map must not return a value, got an object");
            }
            // Verify that no reference to a temporary ndobject was kept
            for (size_t i = 0; i != src_count + 1; ++i) {
                if (ndo[i]->ob_refcnt != 1 || ndo[i]->v.get_ndo()->m_memblockdata.m_use_count != 1) {
                    throw runtime_error("Python callable for elwise_map must not keep a reference "
                                    "to its arguments");
                }
            }

            PyGILState_Release(gstate);
        }

        static void destruct(kernel_data_prefix *extra)
        {
            PyGILState_STATE gstate;
            gstate = PyGILState_Ensure();

            extra_type *e = reinterpret_cast<extra_type *>(extra);
            WNDObject **ndo = reinterpret_cast<WNDObject **>(e + 1);
            size_t src_count = e->src_count;
            Py_XDECREF(e->callable);
            for (size_t i = 0; i != src_count + 1; ++i) {
                Py_XDECREF(ndo[i]);
            }

            PyGILState_Release(gstate);
        }
    };
} // anonymous namespace

class pyobject_elwise_expr_kernel_generator : public expr_kernel_generator {
    pyobject_ownref m_callable;
    dtype m_dst_dt;
    vector<dtype> m_src_dt;
public:
    pyobject_elwise_expr_kernel_generator(PyObject *callable,
                    const dtype& dst_dt, const std::vector<dtype>& src_dt)
        : m_callable(callable, true), m_dst_dt(dst_dt), m_src_dt(src_dt)
    {
    }

    virtual ~pyobject_elwise_expr_kernel_generator() {
    }

    size_t make_expr_kernel(
                assignment_kernel *out, size_t offset_out,
                const dtype& dst_dt, const char *dst_metadata,
                size_t src_count, const dtype *src_dt, const char **src_metadata,
                kernel_request_t kernreq, const eval::eval_context *ectx) const
    {
        if (src_count != m_src_dt.size()) {
            stringstream ss;
            ss << "This elwise_map kernel requires " << m_src_dt.size() << " src operands, ";
            ss << "received " << src_count;
            throw runtime_error(ss.str());
        }
        bool require_elwise = dst_dt != m_dst_dt;
        if (require_elwise) {
            for (size_t i = 0; i != src_count; ++i) {
                if (src_dt[i] != m_src_dt[i]) {
                    require_elwise = true;
                    break;
                }
            }
        }
        // If the dtypes don't match the ones for this generator,
        // call the elementwise dimension handler to handle one dimension,
        // giving 'this' as the next kernel generator to call
        if (require_elwise) {
            return make_elwise_dimension_expr_kernel(out, offset_out,
                            dst_dt, dst_metadata,
                            src_count, src_dt, src_metadata,
                            kernreq, ectx,
                            this);
        }

        size_t extra_size = sizeof(pyobject_expr_kernel_extra) +
                        (src_count + 1) * sizeof(WNDObject *);
        out->ensure_capacity_leaf(offset_out + extra_size);
        pyobject_expr_kernel_extra *e = out->get_at<pyobject_expr_kernel_extra>(offset_out);
        WNDObject **ndo = reinterpret_cast<WNDObject **>(e + 1);
        switch (kernreq) {
            case kernel_request_single:
                e->base.set_function<expr_single_operation_t>(&pyobject_expr_kernel_extra::single);
                break;
            case kernel_request_strided:
                e->base.set_function<expr_strided_operation_t>(&pyobject_expr_kernel_extra::strided);
                break;
            default: {
                stringstream ss;
                ss << "pyobject_elwise_expr_kernel_generator: unrecognized request " << (int)kernreq;
                throw runtime_error(ss.str());
            }
        }
        e->base.destructor = &pyobject_expr_kernel_extra::destruct;
        e->src_count = src_count;
        e->callable = m_callable.get();
        Py_INCREF(e->callable);
        // Create shell WNDObjects which are used to give the kernel data to Python
        strided_dim_dtype_metadata *md;
        dtype dt = make_strided_dim_dtype(dst_dt);
        ndobject n(make_ndobject_memory_block(dt.get_metadata_size()));
        n.get_ndo()->m_dtype = dt.release();
        n.get_ndo()->m_flags = write_access_flag;
        md = reinterpret_cast<strided_dim_dtype_metadata *>(n.get_ndo_meta());
        md->size = 1;
        md->stride = 0;
        if (dst_dt.get_metadata_size() > 0) {
            dst_dt.extended()->metadata_copy_construct(
                            n.get_ndo_meta() + sizeof(strided_dim_dtype_metadata),
                            dst_metadata, NULL);
        }
        ndo[0] = (WNDObject *)wrap_ndobject(DYND_MOVE(n));
        for (size_t i = 0; i != src_count; ++i) {
            dt = make_strided_dim_dtype(src_dt[i]);
            n.set(make_ndobject_memory_block(dt.get_metadata_size()));
            n.get_ndo()->m_dtype = dt.release();
            n.get_ndo()->m_flags = read_access_flag;
            md = reinterpret_cast<strided_dim_dtype_metadata *>(n.get_ndo_meta());
            md->size = 1;
            md->stride = 0;
            if (dst_dt.get_metadata_size() > 0) {
                dst_dt.extended()->metadata_copy_construct(
                                n.get_ndo_meta() + sizeof(strided_dim_dtype_metadata),
                                src_metadata[i], NULL);
            }
            ndo[i+1] = (WNDObject *)wrap_ndobject(DYND_MOVE(n));
        }

        return offset_out + extra_size;
    }
};

PyObject *pydynd::elwise_map(PyObject *n_obj, PyObject *callable,
                PyObject *dst_type, PyObject *src_type)
{
    if (!WNDObject_Check(n_obj)) {
        throw runtime_error("can only call dynd's elwise_map on dynd ndobjects");
    }
    ndobject n = ((WNDObject *)n_obj)->v;
    if (n.get_ndo() == NULL) {
        throw runtime_error("cannot convert NULL dynd ndobject to numpy");
    }

    dtype dst_dt;
    vector<dtype> src_dt;

    dst_dt = make_dtype_from_object(dst_type);
    if (src_type != NULL && src_type != Py_None) {
        // Cast to the source dtype if requested
        src_dt.push_back(make_dtype_from_object(src_type));
        n = n.cast_udtype(src_dt.back());
    } else {
        src_dt.push_back(n.get_udtype());
    }

    size_t undim = n.get_undim();
    dimvector result_shape(undim);
    n.get_shape(result_shape.get());
    dtype result_vdt = dst_dt;
    for (size_t i = 0; i != undim; ++i) {
        if (result_shape[undim - i - 1] == -1) {
            result_vdt = make_var_dim_dtype(result_vdt);
        } else {
            result_vdt = make_strided_dim_dtype(result_vdt);
        }
    }

    // Create the result
    string field_names[1] = {"arg0"};
    ndobject result = combine_into_struct(1, field_names, &n);

    // Because the expr dtype's operand is the result's dtype,
    // we can swap it in as the dtype
    dtype edt = make_expr_dtype(result_vdt,
                    result.get_dtype(),
                    new pyobject_elwise_expr_kernel_generator(callable, dst_dt, src_dt));
    edt.swap(result.get_ndo()->m_dtype);
    return wrap_ndobject(DYND_MOVE(result));
}
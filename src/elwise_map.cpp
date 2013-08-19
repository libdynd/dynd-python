//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>

#include <iostream>

#include <dynd/types/expr_type.hpp>
#include <dynd/types/unary_expr_type.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/types/var_dim_type.hpp>
#include <dynd/kernels/expr_kernel_generator.hpp>
#include <dynd/kernels/elwise_expr_kernels.hpp>
#include <dynd/shape_tools.hpp>

#include "utility_functions.hpp"
#include "elwise_map.hpp"
#include "array_functions.hpp"
#include "type_functions.hpp"

using namespace std;
using namespace dynd;
using namespace pydynd;

namespace {
    struct pyobject_expr_kernel_extra {
        typedef pyobject_expr_kernel_extra extra_type;

        ckernel_prefix base;
        size_t src_count;
        PyObject *callable;
        // After this are 1 + src_count shell WArrays,
        // whose guts are replaced for each call of
        // the kernel

        inline PyObject *set_data_pointers(char *dst, const char * const *src)
        {
            WArray **ndo = reinterpret_cast<WArray **>(this + 1);
            // Modify the temporary arrays to point at the data.
            // The constructor sets the metadata to be a size-1 array,
            // so no need to modify any of it.
            ndo[0]->v.get_ndo()->m_data_pointer = dst;
            for (size_t i = 0; i != src_count; ++i) {
                ndo[i+1]->v.get_ndo()->m_data_pointer = const_cast<char *>(src[i]);
            }

            // Put all the arrays in a tuple
            pyobject_ownref args(PyTuple_New(src_count + 1));
            for (size_t i = 0; i != src_count + 1; ++i) {
                Py_INCREF(ndo[i]);
                PyTuple_SET_ITEM(args.get(), i, (PyObject *)ndo[i]);
            }
            return args.release();
        }

        inline PyObject *set_data_pointers(char *dst, intptr_t dst_stride,
                        const char * const *src, const intptr_t *src_stride,
                        size_t count)
        {
            WArray **ndo = reinterpret_cast<WArray **>(this + 1);
            strided_dim_type_metadata *md;
            // Modify the temporary arrays to point at the data.
            ndo[0]->v.get_ndo()->m_data_pointer = dst;
            md = reinterpret_cast<strided_dim_type_metadata *>(ndo[0]->v.get_ndo_meta());
            md->size = count;
            md->stride = dst_stride;
            for (size_t i = 0; i != src_count; ++i) {
                ndo[i+1]->v.get_ndo()->m_data_pointer = const_cast<char *>(src[i]);
                md = reinterpret_cast<strided_dim_type_metadata *>(ndo[i+1]->v.get_ndo_meta());
                md->size = count;
                md->stride = src_stride[i];
            }

            // Put all the arrays in a tuple
            pyobject_ownref args(PyTuple_New(src_count + 1));
            for (size_t i = 0; i != src_count + 1; ++i) {
                Py_INCREF(ndo[i]);
                PyTuple_SET_ITEM(args.get(), i, (PyObject *)ndo[i]);
            }
            return args.release();
        }

        inline void verify_postcall_consistency(PyObject *res)
        {
            WArray **ndo = reinterpret_cast<WArray **>(this + 1);
            // Verify that nothing was returned
            if (res != Py_None) {
                throw runtime_error("Python callable for elwise_map must not return a value, got an object");
            }
            // Verify that no reference to a temporary array was kept
            for (size_t i = 0; i != src_count + 1; ++i) {
                if (Py_REFCNT(ndo[i]) != 1) {
                    stringstream ss;
                    ss << "The elwise_map callable function held onto a reference to the ";
                    if (i == 0) {
                        ss << "dst";
                    } else {
                        ss << "src_" << i-1 << "";
                    }
                    ss << " argument, this is disallowed";
                    throw runtime_error(ss.str());
                } else if (ndo[i]->v.get_ndo()->m_memblockdata.m_use_count != 1) {
                    stringstream ss;
                    ss << "The elwise_map callable function held onto a reference to the data underlying the ";
                    if (i == 0) {
                        ss << "dst";
                    } else {
                        ss << "src_" << i-1 << "";
                    }
                    ss << " argument, this is disallowed";
                    throw runtime_error(ss.str());
                }
            }
        }

        static void single_unary(char *dst, const char *src,
                        ckernel_prefix *extra)
        {
            PyGILState_RAII pgs;

            extra_type *e = reinterpret_cast<extra_type *>(extra);
            pyobject_ownref args(e->set_data_pointers(dst, &src));
            // Call the function
            pyobject_ownref res(PyObject_Call(e->callable, args.get(), NULL));
            args.clear();
            e->verify_postcall_consistency(res.get());
        }

        static void single(char *dst, const char * const *src,
                        ckernel_prefix *extra)
        {
            PyGILState_RAII pgs;

            extra_type *e = reinterpret_cast<extra_type *>(extra);
            pyobject_ownref args(e->set_data_pointers(dst, src));
            // Call the function
            pyobject_ownref res(PyObject_Call(e->callable, args.get(), NULL));
            args.clear();
            e->verify_postcall_consistency(res.get());
        }

        static void strided_unary(char *dst, intptr_t dst_stride,
                    const char *src, intptr_t src_stride,
                    size_t count, ckernel_prefix *extra)
        {
            PyGILState_RAII pgs;

            extra_type *e = reinterpret_cast<extra_type *>(extra);
            // Put all the arrays in a tuple
            pyobject_ownref args(e->set_data_pointers(dst, dst_stride, &src, &src_stride, count));
            // Call the function
            pyobject_ownref res(PyObject_Call(e->callable, args.get(), NULL));
            args.clear();
            e->verify_postcall_consistency(res.get());
        }

        static void strided(char *dst, intptr_t dst_stride,
                    const char * const *src, const intptr_t *src_stride,
                    size_t count, ckernel_prefix *extra)
        {
            PyGILState_RAII pgs;

            extra_type *e = reinterpret_cast<extra_type *>(extra);
            // Put all the arrays in a tuple
            pyobject_ownref args(e->set_data_pointers(dst, dst_stride, src, src_stride, count));
            // Call the function
            pyobject_ownref res(PyObject_Call(e->callable, args.get(), NULL));
            args.clear();
            e->verify_postcall_consistency(res.get());
        }

        static void destruct(ckernel_prefix *extra)
        {
            PyGILState_RAII pgs;

            extra_type *e = reinterpret_cast<extra_type *>(extra);
            WArray **ndo = reinterpret_cast<WArray **>(e + 1);
            size_t src_count = e->src_count;
            Py_XDECREF(e->callable);
            for (size_t i = 0; i != src_count + 1; ++i) {
                Py_XDECREF(ndo[i]);
            }
        }
    };
} // anonymous namespace

class pyobject_elwise_expr_kernel_generator : public expr_kernel_generator {
    pyobject_ownref m_callable;
    ndt::type m_dst_tp;
    vector<ndt::type> m_src_tp;
public:
    pyobject_elwise_expr_kernel_generator(PyObject *callable,
                    const ndt::type& dst_tp, const std::vector<ndt::type>& src_tp)
        : expr_kernel_generator(true), m_callable(callable, true),
                        m_dst_tp(dst_tp), m_src_tp(src_tp)
    {
    }

    pyobject_elwise_expr_kernel_generator(PyObject *callable,
                    const ndt::type& dst_tp, const ndt::type& src_tp)
        : expr_kernel_generator(true), m_callable(callable, true),
                        m_dst_tp(dst_tp), m_src_tp(1)
    {
        m_src_tp[0] = src_tp;
    }

    virtual ~pyobject_elwise_expr_kernel_generator() {
    }

    size_t make_expr_kernel(
                ckernel_builder *out, size_t offset_out,
                const ndt::type& dst_tp, const char *dst_metadata,
                size_t src_count, const ndt::type *src_tp, const char **src_metadata,
                kernel_request_t kernreq, const eval::eval_context *ectx) const
    {
        if (src_count != m_src_tp.size()) {
            stringstream ss;
            ss << "This elwise_map kernel requires " << m_src_tp.size() << " src operands, ";
            ss << "received " << src_count;
            throw runtime_error(ss.str());
        }
        bool require_elwise = dst_tp != m_dst_tp;
        if (require_elwise) {
            for (size_t i = 0; i != src_count; ++i) {
                if (src_tp[i] != m_src_tp[i]) {
                    require_elwise = true;
                    break;
                }
            }
        }
        // If the types don't match the ones for this generator,
        // call the elementwise dimension handler to handle one dimension
        // or handle input/output buffering, giving 'this' as the next
        // kernel generator to call
        if (require_elwise) {
            return make_elwise_dimension_expr_kernel(out, offset_out,
                            dst_tp, dst_metadata,
                            src_count, src_tp, src_metadata,
                            kernreq, ectx,
                            this);
        }

        size_t extra_size = sizeof(pyobject_expr_kernel_extra) +
                        (src_count + 1) * sizeof(WArray *);
        out->ensure_capacity_leaf(offset_out + extra_size);
        pyobject_expr_kernel_extra *e = out->get_at<pyobject_expr_kernel_extra>(offset_out);
        WArray **ndo = reinterpret_cast<WArray **>(e + 1);
        switch (kernreq) {
            case kernel_request_single:
                if (src_count == 1) {
                    // Unary kernels are special-cased to be the same as assignment kernels
                    e->base.set_function<unary_single_operation_t>(&pyobject_expr_kernel_extra::single_unary);
                } else {
                    e->base.set_function<expr_single_operation_t>(&pyobject_expr_kernel_extra::single);
                }
                break;
            case kernel_request_strided:
                if (src_count == 1) {
                    // Unary kernels are special-cased to be the same as assignment kernels
                    e->base.set_function<unary_strided_operation_t>(&pyobject_expr_kernel_extra::strided_unary);
                } else {
                    e->base.set_function<expr_strided_operation_t>(&pyobject_expr_kernel_extra::strided);
                }
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
        // Create shell WArrays which are used to give the kernel data to Python
        strided_dim_type_metadata *md;
        ndt::type dt = ndt::make_strided_dim(dst_tp);
        nd::array n(make_array_memory_block(dt.get_metadata_size()));
        n.get_ndo()->m_type = dt.release();
        n.get_ndo()->m_flags = nd::write_access_flag;
        md = reinterpret_cast<strided_dim_type_metadata *>(n.get_ndo_meta());
        md->size = 1;
        md->stride = 0;
        if (dst_tp.get_metadata_size() > 0) {
            dst_tp.extended()->metadata_copy_construct(
                            n.get_ndo_meta() + sizeof(strided_dim_type_metadata),
                            dst_metadata, NULL);
        }
        ndo[0] = (WArray *)wrap_array(DYND_MOVE(n));
        for (size_t i = 0; i != src_count; ++i) {
            dt = ndt::make_strided_dim(src_tp[i]);
            n.set(make_array_memory_block(dt.get_metadata_size()));
            n.get_ndo()->m_type = dt.release();
            n.get_ndo()->m_flags = nd::read_access_flag;
            md = reinterpret_cast<strided_dim_type_metadata *>(n.get_ndo_meta());
            md->size = 1;
            md->stride = 0;
            if (src_tp[i].get_metadata_size() > 0) {
                src_tp[i].extended()->metadata_copy_construct(
                                n.get_ndo_meta() + sizeof(strided_dim_type_metadata),
                                src_metadata[i], NULL);
            }
            ndo[i+1] = (WArray *)wrap_array(DYND_MOVE(n));
        }

        return offset_out + extra_size;
    }

    void print_type(std::ostream& o) const
    {
        PyGILState_RAII pgs;

        PyObject *name_obj = PyObject_GetAttrString(m_callable.get(), "__name__");
        if (name_obj != NULL) {
            pyobject_ownref name(name_obj);
            o << pystring_as_string(name.get());
        } else {
            PyErr_Clear();
            o << "_unnamed";
        }
        o << "(op0";
        for (size_t i = 1; i != m_src_tp.size(); ++i) {
            o << ", op" << i;
        }
        o << ")";
    }
};

static PyObject *unary_elwise_map(PyObject *n_obj, PyObject *callable,
                PyObject *dst_type, PyObject *src_type)
{
    nd::array n = array_from_py(n_obj, 0, false);
    if (n.get_ndo() == NULL) {
        throw runtime_error("elwise_map received a NULL dynd array");
    }

    ndt::type dst_tp, src_tp;

    dst_tp = make_ndt_type_from_pyobject(dst_type);
    if (src_type != Py_None) {
        // Cast to the source type if requested
        src_tp = make_ndt_type_from_pyobject(src_type);
        // Do the ucast in a way to match up the dimensions
        n = n.ucast(src_tp, src_tp.get_ndim());
    } else {
        src_tp = n.get_dtype();
    }

    ndt::type edt = ndt::make_unary_expr(dst_tp, src_tp,
                    new pyobject_elwise_expr_kernel_generator(callable, dst_tp, src_tp.value_type()));
    nd::array result = n.replace_dtype(edt, src_tp.get_ndim());
    return wrap_array(result);
}

static PyObject *general_elwise_map(PyObject *n_list, PyObject *callable,
                PyObject *dst_type, PyObject *src_type_list)
{
    vector<nd::array> n(PyList_Size(n_list));
    for (size_t i = 0; i != n.size(); ++i) {
        n[i] = array_from_py(PyList_GET_ITEM(n_list, i), 0, false);
        if (n[i].get_ndo() == NULL) {
            throw runtime_error("elwise_map received a NULL dynd array");
        }
    }

    ndt::type dst_tp;
    vector<ndt::type> src_tp(n.size());

    dst_tp = make_ndt_type_from_pyobject(dst_type);
    if (src_type_list != Py_None) {
        for (size_t i = 0; i != n.size(); ++i) {
            // Cast to the source type if requested
            src_tp[i] = make_ndt_type_from_pyobject(PyList_GET_ITEM(src_type_list, i));
            n[i] = n[i].ucast(src_tp[i]);
        }
    } else {
        for (size_t i = 0; i != n.size(); ++i) {
            src_tp[i] = n[i].get_dtype();
        }
    }

    size_t undim = 0;
    for (size_t i = 0; i != n.size(); ++i) {
        size_t undim_i = n[i].get_ndim();
        if (undim_i > undim) {
            undim = undim_i;
        }
    }
    dimvector result_shape(undim), tmp_shape(undim);
    for (size_t j = 0; j != undim; ++j) {
        result_shape[j] = 1;
    }
    for (size_t i = 0; i != n.size(); ++i) {
        size_t undim_i = n[i].get_ndim();
        if (undim_i > 0) {
            n[i].get_shape(tmp_shape.get());
            incremental_broadcast(undim, result_shape.get(), undim_i, tmp_shape.get());
        }
    }

    ndt::type result_vdt = dst_tp;
    for (size_t j = 0; j != undim; ++j) {
        if (result_shape[undim - j - 1] == -1) {
            result_vdt = ndt::make_var_dim(result_vdt);
        } else {
            result_vdt = ndt::make_strided_dim(result_vdt);
        }
    }

    // Create the result
    vector<string> field_names(n.size());
    for (size_t i = 0; i != n.size(); ++i) {
        stringstream ss;
        ss << "arg" << i;
        field_names[i] = ss.str();
    }
    nd::array result = combine_into_struct(n.size(), &field_names[0], &n[0]);

    // Because the expr type's operand is the result's type,
    // we can swap it in as the type
    ndt::type edt = ndt::make_expr(result_vdt,
                    result.get_type(),
                    new pyobject_elwise_expr_kernel_generator(callable, dst_tp, src_tp));
    edt.swap(result.get_ndo()->m_type);
    return wrap_array(DYND_MOVE(result));
}

PyObject *pydynd::elwise_map(PyObject *n_obj, PyObject *callable,
                PyObject *dst_type, PyObject *src_type)
{
    if (!PyList_Check(n_obj)) {
        PyErr_SetString(PyExc_TypeError, "First parameter to elwise_map, 'n', "
                        "is not a list of dynd arrays");
        return NULL;
    }
    if (src_type != Py_None) {
        if (!PyList_Check(src_type)) {
            PyErr_SetString(PyExc_TypeError, "Fourth parameter to elwise_map, 'src_type', "
                            "is not a list of dynd types");
            return NULL;
        }
    }
    if (PyList_Size(n_obj) == 1) {
        return unary_elwise_map(PyList_GET_ITEM(n_obj, 0), callable, dst_type,
                        src_type == Py_None ? Py_None : PyList_GET_ITEM(src_type, 0));
    } else {
        return general_elwise_map(n_obj, callable, dst_type, src_type);
    }
}

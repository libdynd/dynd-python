//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <sstream>

#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/memblock/external_memory_block.hpp>
#include <dynd/func/lift_arrfunc.hpp>
#include <dynd/func/lift_reduction_arrfunc.hpp>
#include <dynd/types/arrfunc_type.hpp>
#include <dynd/kernels/ckernel_common_functions.hpp>
#include <dynd/kernels/rolling_ckernel_deferred.hpp>
#include <dynd/kernels/reduction_kernels.hpp>
#include <dynd/kernels/take_ckernel_deferred.hpp>

#include "py_lowlevel_api.hpp"
#include "numpy_ufunc_kernel.hpp"
#include "utility_functions.hpp"
#include "exception_translation.hpp"
#include "ckernel_deferred_from_pyfunc.hpp"

using namespace std;
using namespace dynd;
using namespace pydynd;

namespace {
    dynd::array_preamble *get_array_ptr(WArray *obj)
    {
        return obj->v.get_ndo();
    }

    const dynd::base_type *get_base_type_ptr(WType *obj)
    {
        return obj->v.extended();
    }

    PyObject *array_from_ptr(PyObject *tp, PyObject *ptr, PyObject *owner, PyObject *access)
    {
        try {
            ndt::type d = make_ndt_type_from_pyobject(tp);
            size_t ptr_val = pyobject_as_size_t(ptr);
            uint32_t access_flags = pyarg_strings_to_int(
                            access, "access", nd::read_access_flag,
                                "readwrite", nd::read_access_flag|nd::write_access_flag,
                                "readonly", nd::read_access_flag,
                                "immutable", nd::read_access_flag|nd::immutable_access_flag);
            if (d.get_metadata_size() != 0) {
                stringstream ss;
                ss << "Cannot create a dynd array from a raw pointer with non-empty metadata, type: ";
                ss << d;
                throw runtime_error(ss.str());
            }
            nd::array result(make_array_memory_block(0));
            d.swap(result.get_ndo()->m_type);
            result.get_ndo()->m_data_pointer = reinterpret_cast<char *>(ptr_val);
            memory_block_ptr owner_memblock = make_external_memory_block(owner, &py_decref_function);
            Py_INCREF(owner);
            result.get_ndo()->m_data_reference = owner_memblock.release();
            result.get_ndo()->m_flags = access_flags;
            return wrap_array(DYND_MOVE(result));
        } catch(...) {
            translate_exception();
            return NULL;
        }
    }

    PyObject *
    make_assignment_ckernel(void *out_ckb, intptr_t ckb_offset,
                            PyObject *dst_tp_obj, const void *dst_metadata,
                            PyObject *src_tp_obj, const void *src_metadata,
                            PyObject *funcproto_obj, PyObject *kerntype_obj,
                            PyObject *ectx_obj)
    {
        try {
            ckernel_builder *ckb_ptr = reinterpret_cast<ckernel_builder *>(out_ckb);

            ndt::type dst_tp = make_ndt_type_from_pyobject(dst_tp_obj);
            ndt::type src_tp = make_ndt_type_from_pyobject(src_tp_obj);
            if (dst_metadata == NULL && dst_tp.get_metadata_size() != 0) {
                stringstream ss;
                ss << "Cannot create an assignment kernel independent of metadata with non-empty metadata, type: ";
                ss << dst_tp;
                throw runtime_error(ss.str());
            }
            if (src_metadata == NULL && src_tp.get_metadata_size() != 0) {
                stringstream ss;
                ss << "Cannot create an assignment kernel independent of metadata with non-empty metadata, type: ";
                ss << src_tp;
                throw runtime_error(ss.str());
            }
            string kt = pystring_as_string(kerntype_obj);
            string fp = pystring_as_string(funcproto_obj);
            arrfunc_proto_t funcproto;
            if (fp == "unary") {
                funcproto = unary_operation_funcproto;
            } else if (fp == "expr") {
                funcproto = expr_operation_funcproto;
            } else {
                stringstream ss;
                ss << "Invalid function prototype type ";
                print_escaped_utf8_string(ss, fp);
                throw runtime_error(ss.str());
            }
            kernel_request_t kerntype;
            if (kt == "single") {
                kerntype = kernel_request_single;
            } else if (kt == "strided") {
                kerntype = kernel_request_strided;
            } else {
                stringstream ss;
                ss << "Invalid kernel request type ";
                print_escaped_utf8_string(ss, kt);
                throw runtime_error(ss.str());
            }

            const eval::eval_context *ectx = eval_context_from_pyobj(ectx_obj);

            // If an expr kernel is requested, use an adapter
            if (funcproto == expr_operation_funcproto) {
                ckb_offset = kernels::wrap_unary_as_expr_ckernel(
                                ckb_ptr, ckb_offset, kerntype);
            }

            intptr_t kernel_size = make_assignment_kernel(
                ckb_ptr, ckb_offset, dst_tp,
                reinterpret_cast<const char *>(dst_metadata), src_tp,
                reinterpret_cast<const char *>(src_metadata), kerntype,
                assign_error_default, ectx);

            return PyLong_FromSsize_t(kernel_size);
        } catch(...) {
            translate_exception();
            return NULL;
        }
    }

    PyObject *make_arrfunc_from_assignment(PyObject *dst_tp_obj, PyObject *src_tp_obj,
                PyObject *funcproto_obj, PyObject *errmode_obj)
    {
        try {
            nd::array ckd = nd::empty(ndt::make_arrfunc());
            arrfunc *ckd_ptr = reinterpret_cast<arrfunc *>(ckd.get_readwrite_originptr());

            ndt::type dst_tp = make_ndt_type_from_pyobject(dst_tp_obj);
            ndt::type src_tp = make_ndt_type_from_pyobject(src_tp_obj);
            string fp = pystring_as_string(funcproto_obj);
            arrfunc_proto_t funcproto;
            if (fp == "unary") {
                funcproto = unary_operation_funcproto;
            } else if (fp == "expr") {
                funcproto = expr_operation_funcproto;
            } else if (fp == "binary_predicate") {
                funcproto = binary_predicate_funcproto;
            } else {
                stringstream ss;
                ss << "Invalid function prototype type ";
                print_escaped_utf8_string(ss, fp);
                throw runtime_error(ss.str());
            }
            assign_error_mode errmode = pyarg_error_mode(errmode_obj);
            dynd::make_arrfunc_from_assignment(dst_tp, src_tp, src_tp,
                            funcproto, errmode, *ckd_ptr);

            return wrap_array(ckd);
        } catch(...) {
            translate_exception();
            return NULL;
        }
    }

    PyObject *make_arrfunc_from_property(PyObject *tp_obj, PyObject *propname_obj,
                PyObject *funcproto_obj, PyObject *errmode_obj)
    {
        try {
            nd::array ckd = nd::empty(ndt::make_arrfunc());
            arrfunc *ckd_ptr = reinterpret_cast<arrfunc *>(ckd.get_readwrite_originptr());

            ndt::type tp = make_ndt_type_from_pyobject(tp_obj);
            string propname = pystring_as_string(propname_obj);
            string fp = pystring_as_string(funcproto_obj);
            arrfunc_proto_t funcproto;
            if (fp == "unary") {
                funcproto = unary_operation_funcproto;
            } else if (fp == "expr") {
                funcproto = expr_operation_funcproto;
            } else if (fp == "binary_predicate") {
                funcproto = binary_predicate_funcproto;
            } else {
                stringstream ss;
                ss << "Invalid function prototype type ";
                print_escaped_utf8_string(ss, fp);
                throw runtime_error(ss.str());
            }
            assign_error_mode errmode = pyarg_error_mode(errmode_obj);
            dynd::make_arrfunc_from_property(tp, propname,
                            funcproto, errmode, *ckd_ptr);

            return wrap_array(ckd);
        } catch(...) {
            translate_exception();
            return NULL;
        }
    }


    PyObject *lift_arrfunc(PyObject *ckd, PyObject *types)
    {
        try {
            nd::array out_ckd = nd::empty(ndt::make_arrfunc());
            arrfunc *out_ckd_ptr = reinterpret_cast<arrfunc *>(out_ckd.get_readwrite_originptr());
            // Convert all the input parameters
            if (!WArray_Check(ckd) || ((WArray *)ckd)->v.get_type().get_type_id() != arrfunc_type_id) {
                stringstream ss;
                ss << "ckd must be an nd.array of type ckernel_deferred";
                throw dynd::type_error(ss.str());
            }
            const nd::array& ckd_arr = ((WArray *)ckd)->v;
            vector<ndt::type> types_vec;
            pyobject_as_vector_ndt_type(types, types_vec);
            
            dynd::lift_arrfunc(out_ckd_ptr, ckd_arr, types_vec);

            return wrap_array(out_ckd);
        } catch(...) {
            translate_exception();
            return NULL;
        }
    }

    PyObject *lift_reduction_arrfunc(PyObject *elwise_reduction_obj, PyObject *lifted_type_obj,
                    PyObject *dst_initialization_obj, PyObject *axis_obj, PyObject *keepdims_obj,
                    PyObject *associative_obj, PyObject *commutative_obj,
                    PyObject *right_associative_obj, PyObject *reduction_identity_obj)
    {
        try {
            nd::array out_ckd = nd::empty(ndt::make_arrfunc());
            arrfunc *out_ckd_ptr = reinterpret_cast<arrfunc *>(out_ckd.get_readwrite_originptr());
            // Convert all the input parameters
            if (!WArray_Check(elwise_reduction_obj) ||
                        ((WArray *)elwise_reduction_obj)->v.get_type().get_type_id() != arrfunc_type_id) {
                stringstream ss;
                ss << "elwise_reduction must be an nd.array of type ckernel_deferred";
                throw dynd::type_error(ss.str());
            }
            const nd::array& elwise_reduction = ((WArray *)elwise_reduction_obj)->v;
            const arrfunc *elwise_reduction_ckd =
                            reinterpret_cast<const arrfunc *>(elwise_reduction.get_readonly_originptr());

            nd::array dst_initialization;
            if (WArray_Check(dst_initialization_obj) &&
                        ((WArray *)dst_initialization_obj)->v.get_type().get_type_id() == arrfunc_type_id) {
                dst_initialization = ((WArray *)dst_initialization_obj)->v;;
            } else if (dst_initialization_obj != Py_None) {
                stringstream ss;
                ss << "dst_initialization must be None or an nd.array of type ckernel_deferred";
                throw dynd::type_error(ss.str());
            }

            ndt::type lifted_type = make_ndt_type_from_pyobject(lifted_type_obj);

            // This is the number of dimensions being reduced
            intptr_t reduction_ndim = lifted_type.get_ndim() - elwise_reduction_ckd->data_dynd_types[1].get_ndim();

            shortvector<bool> reduction_dimflags(reduction_ndim);
            if (axis_obj == Py_None) {
                // None means to reduce all axes
                for (intptr_t i = 0; i < reduction_ndim; ++i) {
                    reduction_dimflags[i] = true;
                }
            } else {
                memset(reduction_dimflags.get(), 0, reduction_ndim * sizeof(bool));
                vector<intptr_t> axis_vec;
                pyobject_as_vector_intp(axis_obj, axis_vec, true);
                for (size_t i = 0, i_end = axis_vec.size(); i != i_end; ++i) {
                    intptr_t ax = axis_vec[i];
                    if (ax < -reduction_ndim || ax >= reduction_ndim) {
                        throw axis_out_of_bounds(ax, reduction_ndim);
                    } else if (ax < 0) {
                        ax += reduction_ndim;
                    }
                    reduction_dimflags[ax] = true;
                }
            }

            bool keepdims;
            if (keepdims_obj == Py_True) {
                keepdims = true;
            } else if (keepdims_obj == Py_False) {
                keepdims = false;
            } else {
                throw type_error("keepdims must be either True or False");
            }

            bool associative;
            if (associative_obj == Py_True) {
                associative = true;
            } else if (associative_obj == Py_False) {
                associative = false;
            } else {
                throw type_error("associative must be either True or False");
            }

            bool commutative;
            if (commutative_obj == Py_True) {
                commutative = true;
            } else if (commutative_obj == Py_False) {
                commutative = false;
            } else {
                throw type_error("commutative must be either True or False");
            }
            
            bool right_associative;
            if (right_associative_obj == Py_True) {
                right_associative = true;
            } else if (right_associative_obj == Py_False) {
                right_associative = false;
            } else {
                throw type_error("right_associative must be either True or False");
            }
            
            nd::array reduction_identity;
            if (WArray_Check(reduction_identity_obj)) {
                reduction_identity = ((WArray *)reduction_identity_obj)->v;;
            } else if (reduction_identity_obj != Py_None) {
                stringstream ss;
                ss << "reduction_identity must be None or an nd.array";
                throw dynd::type_error(ss.str());
            }

            dynd::lift_reduction_arrfunc(out_ckd_ptr, elwise_reduction,
                        lifted_type, dst_initialization, keepdims,
                        reduction_ndim, reduction_dimflags.get(),
                        associative, commutative, right_associative,
                        reduction_identity);

            return wrap_array(out_ckd);
        } catch(...) {
            translate_exception();
            return NULL;
        }
    }

    static PyObject *make_rolling_ckernel_deferred(PyObject *dst_tp_obj,
                                                   PyObject *src_tp_obj,
                                                   PyObject *window_op_obj,
                                                   PyObject *window_size_obj)
    {
        ndt::type dst_tp = make_ndt_type_from_pyobject(dst_tp_obj);
        ndt::type src_tp = make_ndt_type_from_pyobject(src_tp_obj);
        if (!WArray_Check(window_op_obj) ||
                    ((WArray *)window_op_obj)->v.get_type().get_type_id() != arrfunc_type_id) {
            stringstream ss;
            ss << "window_op must be an nd.array of type ckernel_deferred";
            throw dynd::type_error(ss.str());
        }
        const nd::array& window_op = ((WArray *)window_op_obj)->v;
        intptr_t window_size = pyobject_as_index(window_size_obj);
        return wrap_array(::make_rolling_ckernel_deferred(
            dst_tp, src_tp, window_op, window_size));
    }

    PyObject *make_builtin_mean1d_ckernel_deferred(PyObject *tp_obj,
                                                    PyObject *minp_obj)
    {
        ndt::type tp = make_ndt_type_from_pyobject(tp_obj);
        intptr_t minp = pyobject_as_index(minp_obj);
        return wrap_array(kernels::make_builtin_mean1d_ckernel_deferred(
            tp.get_type_id(), minp));
    }

    PyObject *make_take_ckernel_deferred(PyObject *dst_tp_obj,
                                         PyObject *src_tp_obj,
                                         PyObject *mask_tp_obj)
    {
        ndt::type dst_tp = make_ndt_type_from_pyobject(dst_tp_obj);
        ndt::type src_tp = make_ndt_type_from_pyobject(src_tp_obj);
        ndt::type mask_tp = make_ndt_type_from_pyobject(mask_tp_obj);
        return wrap_array(
            kernels::make_take_ckernel_deferred(dst_tp, src_tp, mask_tp));
    }

    const py_lowlevel_api_t py_lowlevel_api = {
        0, // version, should increment this every time the struct changes at a release
        &get_array_ptr,
        &get_base_type_ptr,
        &array_from_ptr,
        &make_assignment_ckernel,
        &make_arrfunc_from_assignment,
        &make_arrfunc_from_property,
        &pydynd::numpy_typetuples_from_ufunc,
        &pydynd::ckernel_deferred_from_ufunc,
        &lift_arrfunc,
        &lift_reduction_arrfunc,
        &pydynd::ckernel_deferred_from_pyfunc,
        &make_rolling_ckernel_deferred,
        &make_builtin_mean1d_ckernel_deferred,
        &make_take_ckernel_deferred
    };
} // anonymous namespace

extern "C" const void *dynd_get_py_lowlevel_api()
{
    return reinterpret_cast<const void *>(&py_lowlevel_api);
}

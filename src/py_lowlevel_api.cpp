//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <sstream>

#include <dynd/kernels/assignment_kernels.hpp>
#include <dynd/memblock/external_memory_block.hpp>
#include <dynd/func/elwise.hpp>
#include <dynd/func/lift_reduction_arrfunc.hpp>
#include <dynd/kernels/ckernel_common_functions.hpp>
#include <dynd/func/rolling_arrfunc.hpp>
#include <dynd/kernels/reduction_kernels.hpp>
#include <dynd/func/take_arrfunc.hpp>

#include "py_lowlevel_api.hpp"
#include "numpy_ufunc_kernel.hpp"
#include "utility_functions.hpp"
#include "exception_translation.hpp"
#include "arrfunc_functions.hpp"
#include "arrfunc_from_pyfunc.hpp"
#include "arrfunc_from_instantiate_pyfunc.hpp"

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
      try
      {
        ndt::type d = make_ndt_type_from_pyobject(tp);
        if (d.is_symbolic()) {
          stringstream ss;
          ss << "Cannot create a dynd array with symbolic type " << d;
          throw type_error(ss.str());
        }
        size_t ptr_val = pyobject_as_size_t(ptr);
        uint32_t access_flags = pyarg_strings_to_int(
            access, "access", nd::read_access_flag, "readwrite",
            nd::read_access_flag | nd::write_access_flag, "readonly",
            nd::read_access_flag, "immutable",
            nd::read_access_flag | nd::immutable_access_flag);
        nd::array result(make_array_memory_block(d.get_arrmeta_size()));
        if (d.get_flags() & (type_flag_destructor)) {
          stringstream ss;
          ss << "Cannot view raw memory using dynd type " << d;
          throw type_error(ss.str());
        }
        // Create the nd.array with default-constructed arrmeta, WITHOUT
        // allocating memory blocks for the blockref types.
        if (d.get_arrmeta_size() > 0) {
          d.extended()->arrmeta_default_construct(
              result.get_ndo()->get_arrmeta(), false);
        }
        d.swap(result.get_ndo()->m_type);
        result.get_ndo()->m_data_pointer = reinterpret_cast<char *>(ptr_val);
        memory_block_ptr owner_memblock =
            make_external_memory_block(owner, &py_decref_function);
        Py_INCREF(owner);
        result.get_ndo()->m_data_reference = owner_memblock.release();
        result.get_ndo()->m_flags = access_flags;
        return wrap_array(std::move(result));
      }
      catch (...)
      {
        translate_exception();
        return NULL;
      }
    }

    PyObject *make_assignment_ckernel(void *ckb, intptr_t ckb_offset,
                                      PyObject *dst_tp_obj,
                                      const void *dst_arrmeta,
                                      PyObject *src_tp_obj,
                                      const void *src_arrmeta,
                                      PyObject *kernreq_obj, PyObject *ectx_obj)
    {
      try {
        ckernel_builder<kernel_request_host> *ckb_ptr =
            reinterpret_cast<ckernel_builder<kernel_request_host> *>(ckb);

        ndt::type dst_tp = make_ndt_type_from_pyobject(dst_tp_obj);
        ndt::type src_tp = make_ndt_type_from_pyobject(src_tp_obj);
        if (dst_arrmeta == NULL && dst_tp.get_arrmeta_size() != 0) {
          stringstream ss;
          ss << "Cannot create an assignment kernel independent of arrmeta "
                "with non-empty arrmeta, type: ";
          ss << dst_tp;
          throw runtime_error(ss.str());
        }
        if (src_arrmeta == NULL && src_tp.get_arrmeta_size() != 0) {
          stringstream ss;
          ss << "Cannot create an assignment kernel independent of arrmeta "
                "with non-empty arrmeta, type: ";
          ss << src_tp;
          throw runtime_error(ss.str());
        }
        string kr = pystring_as_string(kernreq_obj);
        kernel_request_t kernreq;
        if (kr == "single") {
          kernreq = kernel_request_single;
        }
        else if (kr == "strided") {
          kernreq = kernel_request_strided;
        }
        else {
          stringstream ss;
          ss << "Invalid kernel request type ";
          print_escaped_utf8_string(ss, kr);
          throw runtime_error(ss.str());
        }

        const eval::eval_context *ectx = eval_context_from_pyobj(ectx_obj);

        intptr_t kernel_size = make_assignment_kernel(
            NULL, NULL, ckb_ptr, ckb_offset, dst_tp,
            reinterpret_cast<const char *>(dst_arrmeta), src_tp,
            reinterpret_cast<const char *>(src_arrmeta), kernreq, ectx,
            nd::array());

        return PyLong_FromSsize_t(kernel_size);
      }
      catch (...) {
        translate_exception();
        return NULL;
      }
    }

    PyObject *make_arrfunc_from_assignment(PyObject *dst_tp_obj,
                                           PyObject *src_tp_obj,
                                           PyObject *errmode_obj)
    {
        try {
            ndt::type dst_tp = make_ndt_type_from_pyobject(dst_tp_obj);
            ndt::type src_tp = make_ndt_type_from_pyobject(src_tp_obj);
            assign_error_mode errmode = pyarg_error_mode(errmode_obj);
            nd::arrfunc af =
                ::make_arrfunc_from_assignment(dst_tp, src_tp, errmode);

            return wrap_array(af);
        } catch(...) {
            translate_exception();
            return NULL;
        }
    }

    PyObject *make_arrfunc_from_property(PyObject *tp_obj,
                                         PyObject *propname_obj)
    {
        try {
            ndt::type tp = make_ndt_type_from_pyobject(tp_obj);
            string propname = pystring_as_string(propname_obj);
            nd::arrfunc af =
                ::make_arrfunc_from_property(tp, propname);

            return wrap_array(af);
        } catch(...) {
            translate_exception();
            return NULL;
        }
    }

    PyObject *lift_arrfunc(PyObject *af)
    {
      try {
        // Convert all the input parameters
        if (!WArray_Check(af) ||
            ((WArray *)af)->v.get_type().get_type_id() != arrfunc_type_id) {
          stringstream ss;
          ss << "af must be an nd.array of type arrfunc";
          throw dynd::type_error(ss.str());
        }
        return wrap_array(dynd::nd::functional::elwise(((WArray *)af)->v));
      }
      catch (...) {
        translate_exception();
        return NULL;
      }
    }

    PyObject *lift_reduction_arrfunc(PyObject *elwise_reduction_obj,
                                     PyObject *lifted_type_obj,
                                     PyObject *dst_initialization_obj,
                                     PyObject *axis_obj, PyObject *keepdims_obj,
                                     PyObject *associative_obj,
                                     PyObject *commutative_obj,
                                     PyObject *right_associative_obj,
                                     PyObject *reduction_identity_obj)
    {
      try {
        // Convert all the input parameters
        if (!WArray_Check(elwise_reduction_obj) ||
            ((WArray *)elwise_reduction_obj)->v.get_type().get_type_id() !=
                arrfunc_type_id) {
          stringstream ss;
          ss << "elwise_reduction must be an nd.array of type arrfunc";
          throw dynd::type_error(ss.str());
        }
        const nd::array &elwise_reduction = ((WArray *)elwise_reduction_obj)->v;
        const arrfunc_type_data *elwise_reduction_af =
            reinterpret_cast<const arrfunc_type_data *>(
                elwise_reduction.get_readonly_originptr());
        const arrfunc_type *elwise_reduction_af_tp =
            elwise_reduction.get_type().extended<arrfunc_type>();

        nd::array dst_initialization;
        if (WArray_Check(dst_initialization_obj) &&
            ((WArray *)dst_initialization_obj)->v.get_type().get_type_id() ==
                arrfunc_type_id) {
          dst_initialization = ((WArray *)dst_initialization_obj)->v;
          ;
        }
        else if (dst_initialization_obj != Py_None) {
          stringstream ss;
          ss << "dst_initialization must be None or an nd.array of type "
                "arrfunc";
          throw dynd::type_error(ss.str());
        }

        ndt::type lifted_type = make_ndt_type_from_pyobject(lifted_type_obj);

        // This is the number of dimensions being reduced
        intptr_t reduction_ndim =
            lifted_type.get_ndim() -
            elwise_reduction_af_tp->get_pos_type(0).get_ndim();

        shortvector<bool> reduction_dimflags(reduction_ndim);
        if (axis_obj == Py_None) {
          // None means to reduce all axes
          for (intptr_t i = 0; i < reduction_ndim; ++i) {
            reduction_dimflags[i] = true;
          }
        }
        else {
          memset(reduction_dimflags.get(), 0, reduction_ndim * sizeof(bool));
          vector<intptr_t> axis_vec;
          pyobject_as_vector_intp(axis_obj, axis_vec, true);
          for (size_t i = 0, i_end = axis_vec.size(); i != i_end; ++i) {
            intptr_t ax = axis_vec[i];
            if (ax < -reduction_ndim || ax >= reduction_ndim) {
              throw axis_out_of_bounds(ax, reduction_ndim);
            }
            else if (ax < 0) {
              ax += reduction_ndim;
            }
            reduction_dimflags[ax] = true;
          }
        }

        bool keepdims;
        if (keepdims_obj == Py_True) {
          keepdims = true;
        }
        else if (keepdims_obj == Py_False) {
          keepdims = false;
        }
        else {
          throw type_error("keepdims must be either True or False");
        }

        bool associative;
        if (associative_obj == Py_True) {
          associative = true;
        }
        else if (associative_obj == Py_False) {
          associative = false;
        }
        else {
          throw type_error("associative must be either True or False");
        }

        bool commutative;
        if (commutative_obj == Py_True) {
          commutative = true;
        }
        else if (commutative_obj == Py_False) {
          commutative = false;
        }
        else {
          throw type_error("commutative must be either True or False");
        }

        bool right_associative;
        if (right_associative_obj == Py_True) {
          right_associative = true;
        }
        else if (right_associative_obj == Py_False) {
          right_associative = false;
        }
        else {
          throw type_error("right_associative must be either True or False");
        }

        nd::array reduction_identity;
        if (WArray_Check(reduction_identity_obj)) {
          reduction_identity = ((WArray *)reduction_identity_obj)->v;
          ;
        }
        else if (reduction_identity_obj != Py_None) {
          stringstream ss;
          ss << "reduction_identity must be None or an nd.array";
          throw dynd::type_error(ss.str());
        }

        nd::arrfunc out_af = ::lift_reduction_arrfunc(
            elwise_reduction, lifted_type, dst_initialization, keepdims,
            reduction_ndim, reduction_dimflags.get(), associative, commutative,
            right_associative, reduction_identity);

        return wrap_array(out_af);
      }
      catch (...) {
        translate_exception();
        return NULL;
      }
    }

    PyObject *arrfunc_from_pyfunc(PyObject *pyfunc, PyObject *proto)
    {
        try {
            return wrap_array(pydynd::arrfunc_from_pyfunc(pyfunc, proto));
        } catch(...) {
            translate_exception();
            return NULL;
        }
    }

    static PyObject *make_rolling_arrfunc(PyObject *window_op_obj,
                                          PyObject *window_size_obj)
    {
        try {
            if (!WArrFunc_Check(window_op_obj)) {
                stringstream ss;
                ss << "window_op must be an nd.arrfunc";
                throw dynd::type_error(ss.str());
            }
            const nd::arrfunc& window_op = ((WArrFunc *)window_op_obj)->v;
            intptr_t window_size = pyobject_as_index(window_size_obj);
            return wrap_array(::make_rolling_arrfunc(window_op, window_size));
        } catch(...) {
            translate_exception();
            return NULL;
        }
    }

    PyObject *make_builtin_mean1d_arrfunc(PyObject *tp_obj, PyObject *minp_obj)
    {
        try {
            ndt::type tp = make_ndt_type_from_pyobject(tp_obj);
            intptr_t minp = pyobject_as_index(minp_obj);
            return wrap_array(kernels::make_builtin_mean1d_arrfunc(
                tp.get_type_id(), minp));
        } catch(...) {
            translate_exception();
            return NULL;
        }
    }

    PyObject *make_take_arrfunc()
    {
        try {
            return wrap_array(
                kernels::make_take_arrfunc());
        } catch(...) {
            translate_exception();
            return NULL;
        }
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
        &pydynd::arrfunc_from_ufunc,
        &lift_arrfunc,
        &lift_reduction_arrfunc,
        &arrfunc_from_pyfunc,
        &pydynd::arrfunc_from_instantiate_pyfunc,
        &make_rolling_arrfunc,
        &make_builtin_mean1d_arrfunc,
        &make_take_arrfunc
    };
} // anonymous namespace

extern "C" const void *dynd_get_py_lowlevel_api()
{
    return reinterpret_cast<const void *>(&py_lowlevel_api);
}

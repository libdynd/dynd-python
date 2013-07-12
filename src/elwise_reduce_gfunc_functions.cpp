//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>

#include <algorithm>

#include <dynd/memblock/external_memory_block.hpp>

#include "elwise_reduce_gfunc_functions.hpp"
#include "array_functions.hpp"
#include "utility_functions.hpp"
#include "ctypes_interop.hpp"

using namespace std;
using namespace dynd;
using namespace pydynd;

static void create_elwise_reduce_gfunc_kernel_from_ctypes(dynd::codegen_cache& cgcache,
            PyCFuncPtrObject *cfunc, bool associative, bool commutative, const nd::array& identity,
            dynd::gfunc::elwise_reduce_kernel& out_kernel)
{
#if 0 // TODO reenable
    ndt::type& returntype = out_kernel.m_returntype;
    vector<ndt::type> &paramtypes = out_kernel.m_paramtypes;
    get_ctypes_signature(cfunc, returntype, paramtypes);

    out_kernel.m_associative = associative;
    out_kernel.m_commutative = commutative;

    memory_block_ptr ctypes_memblock = make_external_memory_block(cfunc, &py_decref_function);
    Py_INCREF(cfunc);

    if (returntype.get_type_id() == void_type_id) {
        // TODO: Should support this if the kernel is flagged as commutative,
        //       in which case the first parameter must be an inout pointer parameter
        throw std::runtime_error("Cannot construct a gfunc reduce kernel from a single ctypes function which returns void");
    }

    if (paramtypes.size() == 1) {
        if (!commutative) {
            throw runtime_error("To use an in-place reduction kernel, the kernel must either be commutative, or"
                        " both left and right associative variants must be provided");
        }

    } else if (paramtypes.size() == 2) {
        if (returntype != paramtypes[0] || returntype != paramtypes[1]) {
            std::stringstream ss;
            ss << "A binary reduction kernel must have all three types equal.";
            ss << " Provided signature " << returntype << " (" << paramtypes[0] << ", " << paramtypes[1] << ")";
            throw std::runtime_error(ss.str());
        }
        cgcache.codegen_left_associative_binary_reduce_function_adapter(returntype, get_ctypes_calling_convention(cfunc),
                            *(void **)cfunc->b_ptr, ctypes_memblock.get(), out_kernel.m_left_associative_reduction_kernel);
        if (!commutative) {
            cgcache.codegen_right_associative_binary_reduce_function_adapter(returntype, get_ctypes_calling_convention(cfunc),
                                *(void **)cfunc->b_ptr, ctypes_memblock.get(), out_kernel.m_right_associative_reduction_kernel);
        }

        // The adapted reduction signature has just one parameter
        paramtypes.pop_back();
    } else {
        std::stringstream ss;
        ss << "A single function provided as a gfunc reduce kernel must be binary, the provided one has " << paramtypes.size();
        throw std::runtime_error(ss.str());
    }

    // If an identity is provided, get an immutable version of it as the reduction type
    if (!identity.empty()) {
        out_kernel.m_identity = identity.cast_scalars(returntype).eval_immutable();
    } else {
        out_kernel.m_identity = nd::array();
    }
#endif // TODO reenable
}

void pydynd::elwise_reduce_gfunc_add_kernel(dynd::gfunc::elwise_reduce& gf, dynd::codegen_cache& cgcache, PyObject *kernel,
                            bool associative, bool commutative, const dynd::nd::array& identity)
{
#if 0 // TODO reenable
    if (PyObject_IsSubclass((PyObject *)Py_TYPE(kernel), ctypes.PyCFuncPtrType_Type)) {
        gfunc::elwise_reduce_kernel ergk;

        create_elwise_reduce_gfunc_kernel_from_ctypes(cgcache, (PyCFuncPtrObject *)kernel,
                        associative, commutative, identity, ergk);
        gf.add_kernel(ergk);
        return;
    }

    throw std::runtime_error("Object could not be used as a gfunc kernel");
#endif // TODO reenable
}


PyObject *pydynd::elwise_reduce_gfunc_call(dynd::gfunc::elwise_reduce& gf, PyObject *args, PyObject *kwargs)
{
#if 0 // TODO reenable
    Py_ssize_t nargs = PySequence_Size(args);
    if (nargs == 1) {
        pyobject_ownref arg0_obj(PySequence_GetItem(args, 0));
        nd::array arg0;
        array_init_from_pyobject(arg0, arg0_obj);
        int ndim = arg0.get_type().get_ndim();

        shortvector<dynd_bool> reduce_axes(ndim);

        // axis=[integer OR tuple of integers]
        int axis_count = pyarg_axis_argument(PyDict_GetItemString(kwargs, "axis"), ndim, reduce_axes.get());

        // associate=['left' OR 'right']
        bool rightassoc = pyarg_strings_to_int(PyDict_GetItemString(kwargs, "associate"), "associate", 0,
                            "left", 0,
                            "right", 1) == 1;

        // keepdims
        bool keepdims = pyarg_bool(PyDict_GetItemString(kwargs, "keepdims"), "keepdims", false);

        vector<ndt::type> argtypes(1);
        argtypes[0] = arg0.get_type().value_type();
        const gfunc::elwise_reduce_kernel *ergk = gf.find_matching_kernel(argtypes);
        if (ergk != NULL) {
            if (axis_count > 1 && !ergk->m_commutative) {
                stringstream ss;
                ss << "Cannot call non-commutative reduce gfunc " << gf.get_name() << " with more than one axis";
                throw runtime_error(ss.str());
            }
            throw std::runtime_error("pydynd::elwise_reduce_gfunc_call isn't implemented presently");
            /*
            nd::array result(make_elwise_reduce_kernel_node_copy_kernel(
                        ergk->m_returntype, arg0.get_node(), reduce_axes.get(), rightassoc, keepdims, ergk->m_identity.get_node(),
                        (!rightassoc || ergk->m_commutative) ? ergk->m_left_associative_reduction_kernel :
                                ergk->m_right_associative_reduction_kernel));
            pyobject_ownref result_obj(WArray_Type->tp_alloc(WArray_Type, 0));
            ((WArray *)result_obj.get())->v.swap(result);
            return result_obj.release();
            */
        } else {
            std::stringstream ss;
            ss << gf.get_name() << ": could not find a gfunc kernel matching input type (" << argtypes[0] << ")";
            throw std::runtime_error(ss.str());
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Elementwise reduction gfuncs only support 1 argument");
        return NULL;
    }
#endif // TODO reenable
        PyErr_SetString(PyExc_TypeError, "Elementwise reduction gfuncs disabled presently");
        return NULL;
}


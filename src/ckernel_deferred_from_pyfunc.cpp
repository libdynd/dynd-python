//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>

#include <vector>

#include <dynd/array.hpp>
#include <dynd/types/ckernel_deferred_type.hpp>

#include <array_functions.hpp>
#include <utility_functions.hpp>
#include <ckernel_deferred_from_pyfunc.hpp>

using namespace std;
using namespace dynd;
using namespace pydynd;

namespace {
    struct pyfunc_ckernel_deferred_data {
        // Callable provided from python
        PyObject *instantiate_pyfunc;
        // nd::array of types
        PyObject *types;
        intptr_t data_types_size;
    };

    static void delete_pyfunc_ckernel_deferred_data(void *self_data_ptr)
    {
        PyGILState_RAII pgs;
        pyfunc_ckernel_deferred_data *data =
                        reinterpret_cast<pyfunc_ckernel_deferred_data *>(self_data_ptr);
        Py_XDECREF(data->instantiate_pyfunc);
        Py_XDECREF(data->types);
        free(data);
    }

    static intptr_t instantiate_pyfunc_ckernel_deferred_data(
        void *self_data_ptr, dynd::ckernel_builder *out_ckb,
        intptr_t ckb_offset, const char *const *dynd_metadata,
        uint32_t kerntype, const eval::eval_context *ectx)
    {
        PyGILState_RAII pgs;
        pyfunc_ckernel_deferred_data *data =
                        reinterpret_cast<pyfunc_ckernel_deferred_data *>(self_data_ptr);

        // Turn the out_ckb pointer into an integer
        pyobject_ownref out_ckb_obj(PyLong_FromSize_t(reinterpret_cast<size_t>(out_ckb)));
        pyobject_ownref ckb_offset_obj(PyLong_FromSsize_t(ckb_offset));

        // Turn the metadata pointers into integers to pass them to python
        pyobject_ownref meta(PyTuple_New(data->data_types_size));
        for (intptr_t i = 0; i < data->data_types_size; ++i) {
            PyTuple_SET_ITEM(meta.get(), i, PyLong_FromSize_t(reinterpret_cast<size_t>(dynd_metadata[i])));
        }

        // Turn the kernel request type into a string
        pyobject_ownref kerntype_str;
        if (kerntype == kernel_request_single) {
            kerntype_str.reset(pystring_from_string("single"));
        } else if (kerntype == kernel_request_strided) {
            kerntype_str.reset(pystring_from_string("strided"));
        } else {
            throw runtime_error("unrecognized kernel request type");
        }

        // Copy the evaluation context into a WEvalContext object
        pyobject_ownref ectx_obj(wrap_eval_context(ectx));

        pyobject_ownref args(PyTuple_New(6));
        PyTuple_SET_ITEM(args.get(), 0, out_ckb_obj.release());
        PyTuple_SET_ITEM(args.get(), 1, ckb_offset_obj.release());
        PyTuple_SET_ITEM(args.get(), 2, data->types);
        Py_INCREF(data->types);
        PyTuple_SET_ITEM(args.get(), 3, meta.release());
        PyTuple_SET_ITEM(args.get(), 4, kerntype_str.release());
        PyTuple_SET_ITEM(args.get(), 5, ectx_obj.release());

        pyobject_ownref result_obj(PyObject_Call(data->instantiate_pyfunc, args.get(), NULL));
        intptr_t result = PyLong_AsSsize_t(result_obj);
        if (result < 0) {
            if (PyErr_Occurred()) {
                // Propagate error
                throw exception();
            } else {
                throw runtime_error("invalid value returned from pyfunc ckernel_deferred instantiate");
            }
        }
        return result;
    }
}

PyObject *pydynd::ckernel_deferred_from_pyfunc(PyObject *instantiate_pyfunc, PyObject *types)
{
    nd::array out_ckd = nd::empty(ndt::make_ckernel_deferred());
    ckernel_deferred *out_ckd_ptr = reinterpret_cast<ckernel_deferred *>(out_ckd.get_readwrite_originptr());

    vector<ndt::type> types_vec;
    pyobject_as_vector_ndt_type(types, types_vec);
    nd::array types_arr(types_vec);
    
    out_ckd_ptr->ckernel_funcproto = expr_operation_funcproto;
    out_ckd_ptr->free_func = &delete_pyfunc_ckernel_deferred_data;
    out_ckd_ptr->data_types_size = types_vec.size();
    out_ckd_ptr->data_dynd_types = reinterpret_cast<const ndt::type *>(types_arr.get_readonly_originptr());
    out_ckd_ptr->data_ptr = malloc(sizeof(pyfunc_ckernel_deferred_data));
    out_ckd_ptr->instantiate_func = &instantiate_pyfunc_ckernel_deferred_data;
    pyfunc_ckernel_deferred_data *data_ptr =
                    reinterpret_cast<pyfunc_ckernel_deferred_data *>(out_ckd_ptr->data_ptr);
    data_ptr->data_types_size = types_vec.size();
    data_ptr->instantiate_pyfunc = instantiate_pyfunc;
    Py_INCREF(instantiate_pyfunc);
    data_ptr->types = wrap_array(types_arr);

    return wrap_array(out_ckd);
}

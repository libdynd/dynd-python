//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>

#include <vector>

#include <dynd/array.hpp>
#include <dynd/types/arrfunc_type.hpp>

#include <array_functions.hpp>
#include <utility_functions.hpp>
#include <arrfunc_from_pyfunc.hpp>
#include <type_functions.hpp>

using namespace std;
using namespace dynd;
using namespace pydynd;

namespace {
    struct pyfunc_arrfunc_data {
        // Callable provided from python
        PyObject *instantiate_pyfunc;
        // nd::array of types
        PyObject *types;
        intptr_t data_types_size;
    };

    static void delete_pyfunc_arrfunc_data(void *self_data_ptr)
    {
        PyGILState_RAII pgs;
        pyfunc_arrfunc_data *data =
                        reinterpret_cast<pyfunc_arrfunc_data *>(self_data_ptr);
        Py_XDECREF(data->instantiate_pyfunc);
        Py_XDECREF(data->types);
        free(data);
    }

    static intptr_t instantiate_pyfunc_arrfunc_data(
        void *self_data_ptr, dynd::ckernel_builder *ckb, intptr_t ckb_offset,
        const ndt::type &dst_tp, const char *dst_arrmeta,
        const ndt::type *src_tp, const char *const *src_arrmeta,
        uint32_t kernreq, const eval::eval_context *ectx)
    {
        PyGILState_RAII pgs;
        pyfunc_arrfunc_data *data =
                        reinterpret_cast<pyfunc_arrfunc_data *>(self_data_ptr);

        // Turn the ckb pointer into an integer
        pyobject_ownref ckb_obj(PyLong_FromSize_t(reinterpret_cast<size_t>(ckb)));
        pyobject_ownref ckb_offset_obj(PyLong_FromSsize_t(ckb_offset));

        // Destination type/arrmeta
        pyobject_ownref dst_tp_obj(wrap_ndt_type(dst_tp));
        pyobject_ownref dst_arrmeta_obj(
            PyLong_FromSize_t(reinterpret_cast<size_t>(dst_arrmeta)));

        // Source types/arrmeta
        pyobject_ownref src_tp_obj(PyTuple_New(data->data_types_size - 1));
        for (intptr_t i = 0; i < data->data_types_size - 1; ++i) {
            PyTuple_SET_ITEM(src_tp_obj.get(), i, wrap_ndt_type(src_tp[i]));
        }
        pyobject_ownref src_arrmeta_obj(PyTuple_New(data->data_types_size - 1));
        for (intptr_t i = 0; i < data->data_types_size - 1; ++i) {
            PyTuple_SET_ITEM(
                src_arrmeta_obj.get(), i,
                PyLong_FromSize_t(reinterpret_cast<size_t>(src_arrmeta[i])));
        }

        // Turn the kernel request type into a string
        pyobject_ownref kernreq_obj;
        if (kernreq == kernel_request_single) {
            kernreq_obj.reset(pystring_from_string("single"));
        } else if (kernreq == kernel_request_strided) {
            kernreq_obj.reset(pystring_from_string("strided"));
        } else {
            throw runtime_error("unrecognized kernel request type");
        }

        // Copy the evaluation context into a WEvalContext object
        pyobject_ownref ectx_obj(wrap_eval_context(ectx));

        pyobject_ownref args(PyTuple_New(8));
        PyTuple_SET_ITEM(args.get(), 0, ckb_obj.release());
        PyTuple_SET_ITEM(args.get(), 1, ckb_offset_obj.release());
        PyTuple_SET_ITEM(args.get(), 2, dst_tp_obj.release());
        PyTuple_SET_ITEM(args.get(), 3, dst_arrmeta_obj.release());
        PyTuple_SET_ITEM(args.get(), 4, src_tp_obj.release());
        PyTuple_SET_ITEM(args.get(), 5, src_arrmeta_obj.release());
        PyTuple_SET_ITEM(args.get(), 6, kernreq_obj.release());
        PyTuple_SET_ITEM(args.get(), 7, ectx_obj.release());

        pyobject_ownref result_obj(PyObject_Call(data->instantiate_pyfunc, args.get(), NULL));
        intptr_t result = PyLong_AsSsize_t(result_obj);
        if (result < 0) {
            if (PyErr_Occurred()) {
                // Propagate error
                throw exception();
            } else {
                throw runtime_error("invalid value returned from pyfunc arrfunc instantiate");
            }
        }
        return result;
    }
}

PyObject *pydynd::arrfunc_from_pyfunc(PyObject *instantiate_pyfunc, PyObject *types)
{
    nd::array out_af = nd::empty(ndt::make_arrfunc());
    arrfunc_type_data *out_af_ptr =
        reinterpret_cast<arrfunc_type_data *>(out_af.get_readwrite_originptr());

    vector<ndt::type> types_vec;
    pyobject_as_vector_ndt_type(types, types_vec);
    nd::array types_arr(types_vec);
    
    out_af_ptr->ckernel_funcproto = expr_operation_funcproto;
    out_af_ptr->free_func = &delete_pyfunc_arrfunc_data;
    out_af_ptr->data_types_size = types_vec.size();
    out_af_ptr->data_dynd_types = reinterpret_cast<const ndt::type *>(types_arr.get_readonly_originptr());
    out_af_ptr->data_ptr = malloc(sizeof(pyfunc_arrfunc_data));
    out_af_ptr->instantiate_func = &instantiate_pyfunc_arrfunc_data;
    pyfunc_arrfunc_data *data_ptr =
                    reinterpret_cast<pyfunc_arrfunc_data *>(out_af_ptr->data_ptr);
    data_ptr->data_types_size = types_vec.size();
    data_ptr->instantiate_pyfunc = instantiate_pyfunc;
    Py_INCREF(instantiate_pyfunc);
    data_ptr->types = wrap_array(types_arr);

    return wrap_array(out_af);
}

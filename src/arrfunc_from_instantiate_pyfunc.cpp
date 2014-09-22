//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>

#include <vector>

#include <dynd/array.hpp>
#include <dynd/func/arrfunc.hpp>

#include <array_functions.hpp>
#include <utility_functions.hpp>
#include <arrfunc_from_instantiate_pyfunc.hpp>
#include <type_functions.hpp>
#include <exception_translation.hpp>

using namespace std;
using namespace dynd;
using namespace pydynd;

namespace {
    static void delete_pyfunc_arrfunc_data(arrfunc_type_data *self_af)
    {
        PyObject *instantiate_pyfunc = *self_af->get_data_as<PyObject *>();
        if (instantiate_pyfunc) {
            PyGILState_RAII pgs;
            Py_DECREF(instantiate_pyfunc);
        }
    }

    static intptr_t instantiate_pyfunc_arrfunc_data(
        const arrfunc_type_data *af_self, dynd::ckernel_builder *ckb,
        intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
        const ndt::type *src_tp, const char *const *src_arrmeta,
        kernel_request_t kernreq, aux_buffer *aux,
        const eval::eval_context *ectx)
    {
        PyGILState_RAII pgs;
        PyObject *instantiate_pyfunc = *af_self->get_data_as<PyObject *>();
        intptr_t param_count = af_self->get_param_count();

        if (aux != NULL) {
          throw invalid_argument("unexpected non-NULL aux value to "
                                 "instantiate_pyfunc_arrfunc_data");
        }

        // Turn the ckb pointer into an integer
        pyobject_ownref ckb_obj(PyLong_FromSize_t(reinterpret_cast<size_t>(ckb)));
        pyobject_ownref ckb_offset_obj(PyLong_FromSsize_t(ckb_offset));

        // Destination type/arrmeta
        pyobject_ownref dst_tp_obj(wrap_ndt_type(dst_tp));
        pyobject_ownref dst_arrmeta_obj(
            PyLong_FromSize_t(reinterpret_cast<size_t>(dst_arrmeta)));

        // Source types/arrmeta
        pyobject_ownref src_tp_obj(PyTuple_New(param_count));
        for (intptr_t i = 0; i < param_count; ++i) {
            PyTuple_SET_ITEM(src_tp_obj.get(), i, wrap_ndt_type(src_tp[i]));
        }
        pyobject_ownref src_arrmeta_obj(PyTuple_New(param_count));
        for (intptr_t i = 0; i < param_count; ++i) {
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

        pyobject_ownref result_obj(PyObject_Call(instantiate_pyfunc, args.get(), NULL));
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

PyObject *pydynd::arrfunc_from_instantiate_pyfunc(PyObject *instantiate_pyfunc,
                                                  PyObject *proto_obj)
{
    try {
        nd::array out_af = nd::empty(ndt::make_arrfunc());
        arrfunc_type_data *out_af_ptr =
            reinterpret_cast<arrfunc_type_data *>(out_af.get_readwrite_originptr());

        ndt::type proto = make_ndt_type_from_pyobject(proto_obj);
        if (proto.get_type_id() != funcproto_type_id) {
            stringstream ss;
            ss << "creating a dynd arrfunc from a python func requires a function "
                  "prototype, was given type " << proto;
            throw type_error(ss.str());
        }
    
        out_af_ptr->free_func = &delete_pyfunc_arrfunc_data;
        out_af_ptr->func_proto = proto;
        *out_af_ptr->get_data_as<PyObject *>() = instantiate_pyfunc;
        Py_INCREF(instantiate_pyfunc);
        out_af_ptr->instantiate = &instantiate_pyfunc_arrfunc_data;

        out_af.flag_as_immutable();
        return wrap_array(out_af);
    } catch(...) {
        translate_exception();
        return NULL;
    }
}

//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>

#include <vector>

#include <dynd/array.hpp>
#include <dynd/types/arrfunc_type.hpp>
#include <dynd/kernels/expr_kernels.hpp>
#include <dynd/func/callable.hpp>

#include <array_functions.hpp>
#include <utility_functions.hpp>
#include <arrfunc_from_pyfunc.hpp>
#include <type_functions.hpp>
#include <exception_translation.hpp>
#include <array_assign_from_py.hpp>

using namespace std;
using namespace dynd;
using namespace pydynd;

namespace {
    struct pyfunc_expr_ck : public kernels::general_ck<pyfunc_expr_ck> {
        // Reference to the python function object
        PyObject *m_pyfunc;
        // The concrete prototype the ckernel is for
        ndt::type m_proto;
        // The arrmeta
        const char *m_dst_arrmeta;
        vector<const char *> m_src_arrmeta;
        eval::eval_context m_ectx;

        inline pyfunc_expr_ck()
            : m_pyfunc(NULL)
        {
        }

        inline ~pyfunc_expr_ck()
        {
            if (m_pyfunc != NULL) {
                PyGILState_RAII pgs;
                Py_DECREF(m_pyfunc);
            }
        }

        /**
         * Initializes just the base.function member
         */
        inline void init_kernfunc(kernel_request_t kernreq)
        {
          base.set_expr_function<self_type>(kernreq);
        }

        inline void verify_postcall_consistency(PyObject *args)
        {
            intptr_t param_count = PyTuple_GET_SIZE(args);
            // Verify that no reference to a temporary array was kept
            for (intptr_t i = 0; i != param_count; ++i) {
                PyObject *item = PyTuple_GET_ITEM(args, i);
                if (Py_REFCNT(item) != 1 ||
                        ((WArray *)item)->v.get_ndo()->m_memblockdata.m_use_count !=
                            1) {
                    stringstream ss;
                    ss << "Python callback function ";
                    pyobject_ownref pyfunc_repr(PyObject_Repr(m_pyfunc));
                    ss << pystring_as_string(pyfunc_repr.get());
                    ss << ", called by dynd, held a reference to parameter ";
                    ss << (i + 1) << " which contained temporary memory.";
                    ss << " This is disallowed.\n";
                    ss << "Python wrapper ref count: " << Py_REFCNT(item) << "\n";
                    ((WArray *)item)->v.debug_print(ss);
                    // Set all the args' data pointers to NULL as a precaution
                    for (i = 0; i != param_count; ++i) {
                        ((WArray *)item)->v.get_ndo()->m_data_pointer = NULL;
                    }
                    throw runtime_error(ss.str());
                }
            }
        }

        static void single(char *dst, char **src,
                           ckernel_prefix *rawself)
        {
            self_type *self = get_self(rawself);
            const funcproto_type *fpt = self->m_proto.tcast<funcproto_type>();
            intptr_t param_count = fpt->get_param_count();
            const ndt::type& dst_tp = fpt->get_return_type();
            const ndt::type *src_tp = fpt->get_param_types_raw();
            // First set up the parameters in a tuple
            pyobject_ownref args(PyTuple_New(param_count));
            for (intptr_t i = 0; i != param_count; ++i) {
                ndt::type tp = src_tp[i];
                nd::array n(make_array_memory_block(tp.get_arrmeta_size()));
                n.get_ndo()->m_type = tp.release();
                n.get_ndo()->m_flags = nd::read_access_flag;
                n.get_ndo()->m_data_pointer = const_cast<char *>(src[i]);
                if (src_tp[i].get_arrmeta_size() > 0) {
                    src_tp[i].extended()->arrmeta_copy_construct(
                        n.get_arrmeta(), self->m_src_arrmeta[i], NULL);
                }
                PyTuple_SET_ITEM(args.get(), i, wrap_array(DYND_MOVE(n)));
            }
            // Now call the function
            pyobject_ownref res(
                PyObject_Call(self->m_pyfunc, args.get(), NULL));
            // Copy the result into the destination memory
            array_no_dim_broadcast_assign_from_py(dst_tp, self->m_dst_arrmeta,
                                                 dst, res.get(), &self->m_ectx);
            res.clear();
            // Validate that the call didn't hang onto the ephemeral data
            // pointers we used. This is done after the dst assignment, because
            // the function result may have contained a reference to an argument.
            self->verify_postcall_consistency(args.get());
        }

        static void strided(char *dst, intptr_t dst_stride,
                            char **src, const intptr_t *src_stride,
                            size_t count, ckernel_prefix *rawself)
        {
            self_type *self = get_self(rawself);
            const funcproto_type *fpt = self->m_proto.tcast<funcproto_type>();
            intptr_t param_count = fpt->get_param_count();
            const ndt::type& dst_tp = fpt->get_return_type();
            const ndt::type *src_tp = fpt->get_param_types_raw();
            // First set up the parameters in a tuple
            pyobject_ownref args(PyTuple_New(param_count));
            for (intptr_t i = 0; i != param_count; ++i) {
                ndt::type tp = src_tp[i];
                nd::array n(make_array_memory_block(tp.get_arrmeta_size()));
                n.get_ndo()->m_type = tp.release();
                n.get_ndo()->m_flags = nd::read_access_flag;
                n.get_ndo()->m_data_pointer = const_cast<char *>(src[i]);
                if (src_tp[i].get_arrmeta_size() > 0) {
                    src_tp[i].extended()->arrmeta_copy_construct(
                        n.get_arrmeta(), self->m_src_arrmeta[i], NULL);
                }
                PyTuple_SET_ITEM(args.get(), i, wrap_array(DYND_MOVE(n)));
            }
            // Do the loop, reusing the args we created
            for (size_t j = 0; j != count; ++j) {
                // Call the function
                pyobject_ownref res(
                    PyObject_Call(self->m_pyfunc, args.get(), NULL));
                // Copy the result into the destination memory
                array_no_dim_broadcast_assign_from_py(
                    dst_tp, self->m_dst_arrmeta, dst, res.get(), &self->m_ectx);
                res.clear();
                // Validate that the call didn't hang onto the ephemeral data
                // pointers we used. This is done after the dst assignment, because
                // the function result may have contained a reference to an argument.
                self->verify_postcall_consistency(args.get());
                // Increment to the next one
                dst += dst_stride;
                for (intptr_t i = 0; i != param_count; ++i) {
                    const nd::array& n = ((WArray *)PyTuple_GET_ITEM(args.get(), i))->v;
                    n.get_ndo()->m_data_pointer += src_stride[i];
                }
            }
        }
    };

    static void delete_arrfunc_data(arrfunc_type_data *self_af)
    {
        PyObject *pyfunc = *self_af->get_data_as<PyObject *>();
        if (pyfunc) {
            PyGILState_RAII pgs;
            Py_DECREF(pyfunc);
        }
    }

    static intptr_t instantiate_arrfunc_data(
        const arrfunc_type_data *af_self, dynd::ckernel_builder *ckb,
        intptr_t ckb_offset, const ndt::type &dst_tp, const char *dst_arrmeta,
        const ndt::type *src_tp, const char *const *src_arrmeta,
        kernel_request_t kernreq, const nd::array &aux,
        const eval::eval_context *ectx)
    {
        typedef pyfunc_expr_ck self_type;
        PyGILState_RAII pgs;
        intptr_t param_count = af_self->get_param_count();

        if (!aux.is_null()) {
          throw invalid_argument("unexpected non-NULL aux value to "
                                 "arrfunc_from_pyfunc instantiation");
        }

        self_type *self = self_type::create(ckb, kernreq, ckb_offset);
        self->m_proto = ndt::make_funcproto(param_count, src_tp, dst_tp);
        self->m_pyfunc = *af_self->get_data_as<PyObject *>();
        Py_XINCREF(self->m_pyfunc);
        self->m_dst_arrmeta = dst_arrmeta;
        self->m_src_arrmeta.resize(param_count);
        copy(src_arrmeta, src_arrmeta + param_count,
             self->m_src_arrmeta.begin());
        self->m_ectx = *ectx;
        return ckb_offset;
    }
}

void pydynd::arrfunc_from_pyfunc(arrfunc_type_data *out_af,
                                 PyObject *instantiate_pyfunc,
                                 const ndt::type &proto)
{
    if (proto.get_type_id() != funcproto_type_id) {
        stringstream ss;
        ss << "creating a dynd arrfunc from a python func requires a function "
                "prototype, was given type " << proto;
        throw type_error(ss.str());
    }
    
    out_af->free_func = &delete_arrfunc_data;
    out_af->func_proto = proto;
    *out_af->get_data_as<PyObject *>() = instantiate_pyfunc;
    Py_INCREF(instantiate_pyfunc);
    out_af->instantiate = &instantiate_arrfunc_data;
}

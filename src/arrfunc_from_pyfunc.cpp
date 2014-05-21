//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>

#include <vector>

#include <dynd/array.hpp>
#include <dynd/types/arrfunc_type.hpp>
#include <dynd/kernels/expr_kernels.hpp>

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
            switch (kernreq) {
            case kernel_request_single:
                base.set_function<expr_single_operation_t>(&self_type::single);
                break;
            case kernel_request_strided:
                base.set_function<expr_strided_operation_t>(&self_type::strided);
                break;
            default: {
                std::stringstream ss;
                ss << "assignment ckernel init: unrecognized ckernel request " << (int)kernreq;
                throw std::invalid_argument(ss.str());
            }
            }
        }

        inline void verify_postcall_consistency(PyObject *args)
        {
            size_t param_count = PyTuple_GET_SIZE(args);
            // Verify that no reference to a temporary array was kept
            for (size_t i = 0; i != param_count; ++i) {
                PyObject *item = PyTuple_GET_ITEM(args, i);
                if (Py_REFCNT(item) != 1 || ((WArray *)item)->v.get_ndo()->m_memblockdata.m_use_count != 1) {
                    stringstream ss;
                    ss << "Python callback function ";
                    pyobject_ownref pyfunc_repr(PyObject_Repr(m_pyfunc));
                    ss << pystring_as_string(pyfunc_repr.get());
                    ss << ", called by dynd, held a reference to parameter ";
                    ss << (i + 1) << " which contained temporary memory.";
                    ss << " This is disallowed.";
                    // Set all the args' data pointers to NULL as a precaution
                    for (i = 0; i != param_count; ++i) {
                        ((WArray *)item)->v.get_ndo()->m_data_pointer = NULL;
                    }
                    throw runtime_error(ss.str());
                }
            }
        }

        static void single(char *dst, const char *const *src,
                           ckernel_prefix *rawself)
        {
            self_type *self = get_self(rawself);
            const funcproto_type *fpt = self->m_proto.tcast<funcproto_type>();
            size_t param_count = fpt->get_param_count();
            const ndt::type& dst_tp = fpt->get_return_type();
            const ndt::type *src_tp = fpt->get_param_types_raw();
            // First set up the parameters in a tuple
            pyobject_ownref args(PyTuple_New(param_count));
            for (size_t i = 0; i != param_count; ++i) {
                ndt::type tp = src_tp[i];
                nd::array n(make_array_memory_block(tp.get_metadata_size()));
                n.get_ndo()->m_type = tp.release();
                n.get_ndo()->m_flags = nd::read_access_flag;
                n.get_ndo()->m_data_pointer = const_cast<char *>(src[i]);
                if (src_tp[i].get_metadata_size() > 0) {
                    src_tp[i].extended()->metadata_copy_construct(
                        n.get_arrmeta(), self->m_src_arrmeta[i], NULL);
                }
                PyTuple_SET_ITEM(args.get(), i, wrap_array(DYND_MOVE(n)));
            }
            // Now call the function
            pyobject_ownref res(
                PyObject_Call(self->m_pyfunc, args.get(), NULL));
            // Validate that the call didn't hang onto the ephemeral data
            // pointers we used
            self->verify_postcall_consistency(args.get());
            // Copy the result into the destination memory
            array_nodim_broadcast_assign_from_py(dst_tp, self->m_dst_arrmeta,
                                                 dst, res.get());
        }

        static void strided(char *dst, intptr_t dst_stride,
                            const char *const *src, const intptr_t *src_stride,
                            size_t count, ckernel_prefix *rawself)
        {
            self_type *self = get_self(rawself);
            const funcproto_type *fpt = self->m_proto.tcast<funcproto_type>();
            size_t param_count = fpt->get_param_count();
            const ndt::type& dst_tp = fpt->get_return_type();
            const ndt::type *src_tp = fpt->get_param_types_raw();
            // First set up the parameters in a tuple
            pyobject_ownref args(PyTuple_New(param_count));
            for (size_t i = 0; i != param_count; ++i) {
                ndt::type tp = src_tp[i];
                nd::array n(make_array_memory_block(tp.get_metadata_size()));
                n.get_ndo()->m_type = tp.release();
                n.get_ndo()->m_flags = nd::read_access_flag;
                n.get_ndo()->m_data_pointer = const_cast<char *>(src[i]);
                if (src_tp[i].get_metadata_size() > 0) {
                    src_tp[i].extended()->metadata_copy_construct(
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
                array_nodim_broadcast_assign_from_py(dst_tp, self->m_dst_arrmeta,
                                                     dst, res.get());
                res.clear();
                // Validate that the call didn't hang onto the ephemeral data
                // pointers we used. This is done after the dst assignment, because
                // the function result may have contained a reference to an argument.
                self->verify_postcall_consistency(args.get());
                // Increment to the next one
                dst += dst_stride;
                for (size_t i = 0; i != param_count; ++i) {
                    const nd::array& n = ((WArray *)PyTuple_GET_ITEM(args.get(), i))->v;
                    n.get_ndo()->m_data_pointer += src_stride[i];
                }
            }
        }
    };

    static void delete_arrfunc_data(void *self_data_ptr)
    {
        PyObject *pyfunc = reinterpret_cast<PyObject *>(self_data_ptr);
        if (pyfunc) {
            PyGILState_RAII pgs;
            Py_DECREF(pyfunc);
        }
    }

    static intptr_t instantiate_arrfunc_data(
        const arrfunc_type_data *af_self, dynd::ckernel_builder *ckb, intptr_t ckb_offset,
        const ndt::type &dst_tp, const char *dst_arrmeta,
        const ndt::type *src_tp, const char *const *src_arrmeta,
        uint32_t kernreq, const eval::eval_context *DYND_UNUSED(ectx))
    {
        typedef pyfunc_expr_ck self_type;
        PyGILState_RAII pgs;
        intptr_t param_count = af_self->get_param_count();

        self_type *self = self_type::create(ckb, ckb_offset, (kernel_request_t)kernreq);
        intptr_t ckb_end = ckb_offset + sizeof(self_type);
        self->m_proto = ndt::make_funcproto(param_count, src_tp, dst_tp);
        self->m_pyfunc = reinterpret_cast<PyObject *>(af_self->data_ptr);
        Py_XINCREF(self->m_pyfunc);
        self->m_dst_arrmeta = dst_arrmeta;
        self->m_src_arrmeta.resize(param_count);
        copy(src_arrmeta, src_arrmeta + param_count,
             self->m_src_arrmeta.begin());
        return ckb_end;
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
    
    out_af->ckernel_funcproto = expr_operation_funcproto;
    out_af->free_func = &delete_arrfunc_data;
    out_af->func_proto = proto;
    out_af->data_ptr = instantiate_pyfunc;
    Py_INCREF(instantiate_pyfunc);
    out_af->instantiate = &instantiate_arrfunc_data;
}

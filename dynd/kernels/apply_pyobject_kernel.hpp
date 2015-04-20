#pragma once

#include "config.hpp"
#include <dynd/kernels/base_kernel.hpp>

namespace pydynd {
namespace nd {
  namespace functional {

    struct apply_pyobject_kernel
        : base_kernel<apply_pyobject_kernel, kernel_request_host, -1> {
      typedef apply_pyobject_kernel self_type;

      // Reference to the python function object
      PyObject *m_pyfunc;
      // The concrete prototype the ckernel is for
      ndt::type m_proto;
      // The arrmeta
      const char *m_dst_arrmeta;
      std::vector<const char *> m_src_arrmeta;
      eval::eval_context m_ectx;

      apply_pyobject_kernel() : m_pyfunc(NULL) {}

      ~apply_pyobject_kernel()
      {
        if (m_pyfunc != NULL) {
          PyGILState_RAII pgs;
          Py_DECREF(m_pyfunc);
        }
      }

      void verify_postcall_consistency(PyObject *args)
      {
        intptr_t nsrc = PyTuple_GET_SIZE(args);
        // Verify that no reference to a temporary array was kept
        for (intptr_t i = 0; i != nsrc; ++i) {
          PyObject *item = PyTuple_GET_ITEM(args, i);
          if (Py_REFCNT(item) != 1 ||
              ((WArray *)item)->v.get_ndo()->m_memblockdata.m_use_count != 1) {
            std::stringstream ss;
            ss << "Python callback function ";
            pyobject_ownref pyfunc_repr(PyObject_Repr(m_pyfunc));
            ss << pystring_as_string(pyfunc_repr.get());
            ss << ", called by dynd, held a reference to parameter ";
            ss << (i + 1) << " which contained temporary memory.";
            ss << " This is disallowed.\n";
            ss << "Python wrapper ref count: " << Py_REFCNT(item) << "\n";
            ((WArray *)item)->v.debug_print(ss);
            // Set all the args' data pointers to NULL as a precaution
            for (i = 0; i != nsrc; ++i) {
              ((WArray *)item)->v.get_ndo()->m_data_pointer = NULL;
            }
            throw std::runtime_error(ss.str());
          }
        }
      }

      void single(char *dst, char *const *src)
      {
        const ndt::arrfunc_type *fpt = m_proto.extended<ndt::arrfunc_type>();
        intptr_t nsrc = fpt->get_npos();
        const ndt::type &dst_tp = fpt->get_return_type();
        const ndt::type *src_tp = fpt->get_pos_types_raw();
        // First set up the parameters in a tuple
        pyobject_ownref args(PyTuple_New(nsrc));
        for (intptr_t i = 0; i != nsrc; ++i) {
          ndt::type tp = src_tp[i];
          nd::array n(make_array_memory_block(tp.get_arrmeta_size()));
          n.get_ndo()->m_type = tp.release();
          n.get_ndo()->m_flags = nd::read_access_flag;
          n.get_ndo()->m_data_pointer = const_cast<char *>(src[i]);
          if (src_tp[i].get_arrmeta_size() > 0) {
            src_tp[i].extended()->arrmeta_copy_construct(
                n.get_arrmeta(), m_src_arrmeta[i], NULL);
          }
          PyTuple_SET_ITEM(args.get(), i, wrap_array(std::move(n)));
        }
        // Now call the function
        pyobject_ownref res(PyObject_Call(m_pyfunc, args.get(), NULL));
        // Copy the result into the destination memory
        array_no_dim_broadcast_assign_from_py(dst_tp, m_dst_arrmeta, dst,
                                              res.get(), &m_ectx);
        res.clear();
        // Validate that the call didn't hang onto the ephemeral data
        // pointers we used. This is done after the dst assignment, because
        // the function result may have contained a reference to an argument.
        verify_postcall_consistency(args.get());
      }

      void strided(char *dst, intptr_t dst_stride, char *const *src,
                   const intptr_t *src_stride, size_t count)
      {
        const ndt::arrfunc_type *fpt = m_proto.extended<ndt::arrfunc_type>();
        intptr_t nsrc = fpt->get_npos();
        const ndt::type &dst_tp = fpt->get_return_type();
        const ndt::type *src_tp = fpt->get_pos_types_raw();
        // First set up the parameters in a tuple
        pyobject_ownref args(PyTuple_New(nsrc));
        for (intptr_t i = 0; i != nsrc; ++i) {
          ndt::type tp = src_tp[i];
          nd::array n(make_array_memory_block(tp.get_arrmeta_size()));
          n.get_ndo()->m_type = tp.release();
          n.get_ndo()->m_flags = nd::read_access_flag;
          n.get_ndo()->m_data_pointer = const_cast<char *>(src[i]);
          if (src_tp[i].get_arrmeta_size() > 0) {
            src_tp[i].extended()->arrmeta_copy_construct(
                n.get_arrmeta(), m_src_arrmeta[i], NULL);
          }
          PyTuple_SET_ITEM(args.get(), i, wrap_array(std::move(n)));
        }
        // Do the loop, reusing the args we created
        for (size_t j = 0; j != count; ++j) {
          // Call the function
          pyobject_ownref res(PyObject_Call(m_pyfunc, args.get(), NULL));
          // Copy the result into the destination memory
          array_no_dim_broadcast_assign_from_py(dst_tp, m_dst_arrmeta, dst,
                                                res.get(), &m_ectx);
          res.clear();
          // Validate that the call didn't hang onto the ephemeral data
          // pointers we used. This is done after the dst assignment, because
          // the function result may have contained a reference to an argument.
          verify_postcall_consistency(args.get());
          // Increment to the next one
          dst += dst_stride;
          for (intptr_t i = 0; i != nsrc; ++i) {
            const nd::array &n = ((WArray *)PyTuple_GET_ITEM(args.get(), i))->v;
            n.get_ndo()->m_data_pointer += src_stride[i];
          }
        }
      }

      static intptr_t
      instantiate(const arrfunc_type_data *af_self, const ndt::arrfunc_type *af_tp,
                  char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
                  const ndt::type &dst_tp, const char *dst_arrmeta,
                  intptr_t nsrc, const ndt::type *src_tp,
                  const char *const *src_arrmeta, kernel_request_t kernreq,
                  const eval::eval_context *ectx, const nd::array &kwds,
                  const std::map<nd::string, ndt::type> &tp_vars)
      {
        PyGILState_RAII pgs;

        self_type *self = self_type::make(ckb, kernreq, ckb_offset);
        self->m_proto = ndt::make_arrfunc(nsrc, src_tp, dst_tp);
        self->m_pyfunc = *af_self->get_data_as<PyObject *>();
        Py_XINCREF(self->m_pyfunc);
        self->m_dst_arrmeta = dst_arrmeta;
        self->m_src_arrmeta.resize(nsrc);
        copy(src_arrmeta, src_arrmeta + nsrc, self->m_src_arrmeta.begin());
        self->m_ectx = *ectx;
        return ckb_offset;
      }

      static void free(arrfunc_type_data *self_af)
      {
        PyObject *pyfunc = *self_af->get_data_as<PyObject *>();
        if (pyfunc) {
          PyGILState_RAII pgs;
          Py_DECREF(pyfunc);
        }
      }
    };
  } // namespace pydynd::nd::functional
} // namespace pydynd::nd
} // namespace pydynd
#pragma once

#include "config.hpp"
#include <dynd/kernels/base_kernel.hpp>

namespace pydynd {
namespace nd {
  namespace functional {

    struct apply_pyobject_kernel
        : dynd::nd::base_kernel<apply_pyobject_kernel> {
      typedef apply_pyobject_kernel self_type;

      // Reference to the python function object
      PyObject *m_pyfunc;
      // The concrete prototype the ckernel is for
      dynd::ndt::type m_proto;
      // The arrmeta
      const char *m_dst_arrmeta;
      std::vector<const char *> m_src_arrmeta;
      dynd::eval::eval_context m_ectx;

      apply_pyobject_kernel() : m_pyfunc(NULL)
      {
      }

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
              ((DyND_PyArrayObject *)item)->v.get()->m_use_count != 1) {
            std::stringstream ss;
            ss << "Python callback function ";
            pyobject_ownref pyfunc_repr(PyObject_Repr(m_pyfunc));
            ss << pydynd::pystring_as_string(pyfunc_repr.get());
            ss << ", called by dynd, held a reference to parameter ";
            ss << (i + 1) << " which contained temporary memory.";
            ss << " This is disallowed.\n";
            ss << "Python wrapper ref count: " << Py_REFCNT(item) << "\n";
            ((DyND_PyArrayObject *)item)->v.debug_print(ss);
            // Set all the args' data pointers to NULL as a precaution
            for (i = 0; i != nsrc; ++i) {
              ((DyND_PyArrayObject *)item)->v.get()->data = NULL;
            }
            throw std::runtime_error(ss.str());
          }
        }
      }

      void single(char *dst, char *const *src)
      {
        const dynd::ndt::callable_type *fpt =
            m_proto.extended<dynd::ndt::callable_type>();
        intptr_t nsrc = fpt->get_npos();
        const dynd::ndt::type &dst_tp = fpt->get_return_type();
        const dynd::ndt::type *src_tp = fpt->get_pos_types_raw();
        // First set up the parameters in a tuple
        pyobject_ownref args(PyTuple_New(nsrc));
        for (intptr_t i = 0; i != nsrc; ++i) {
          dynd::ndt::type tp = src_tp[i];
          dynd::nd::array n(
              dynd::make_array_memory_block(tp.get_arrmeta_size()));
          n.get()->tp = tp;
          n.get()->flags = dynd::nd::read_access_flag;
          n.get()->data = const_cast<char *>(src[i]);
          if (src_tp[i].get_arrmeta_size() > 0) {
            src_tp[i].extended()->arrmeta_copy_construct(
                n.get()->metadata(), m_src_arrmeta[i],
                dynd::intrusive_ptr<dynd::memory_block_data>());
          }
          PyTuple_SET_ITEM(args.get(), i, DyND_PyWrapper_New(std::move(n)));
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
        const dynd::ndt::callable_type *fpt =
            m_proto.extended<dynd::ndt::callable_type>();
        intptr_t nsrc = fpt->get_npos();
        const dynd::ndt::type &dst_tp = fpt->get_return_type();
        const dynd::ndt::type *src_tp = fpt->get_pos_types_raw();
        // First set up the parameters in a tuple
        pyobject_ownref args(PyTuple_New(nsrc));
        for (intptr_t i = 0; i != nsrc; ++i) {
          dynd::ndt::type tp = src_tp[i];
          dynd::nd::array n(
              dynd::make_array_memory_block(tp.get_arrmeta_size()));
          n.get()->tp = tp;
          n.get()->flags = dynd::nd::read_access_flag;
          n.get()->data = const_cast<char *>(src[i]);
          if (src_tp[i].get_arrmeta_size() > 0) {
            src_tp[i].extended()->arrmeta_copy_construct(
                n.get()->metadata(), m_src_arrmeta[i],
                dynd::intrusive_ptr<dynd::memory_block_data>());
          }
          PyTuple_SET_ITEM(args.get(), i, DyND_PyWrapper_New(std::move(n)));
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
            const dynd::nd::array &n =
                ((DyND_PyArrayObject *)PyTuple_GET_ITEM(args.get(), i))->v;
            n.get()->data += src_stride[i];
          }
        }
      }

      static intptr_t instantiate(
          char *static_data, size_t DYND_UNUSED(data_size),
          char *DYND_UNUSED(data), void *ckb, intptr_t ckb_offset,
          const dynd::ndt::type &dst_tp, const char *dst_arrmeta, intptr_t nsrc,
          const dynd::ndt::type *src_tp, const char *const *src_arrmeta,
          dynd::kernel_request_t kernreq, const dynd::eval::eval_context *ectx,
          intptr_t nkwd, const dynd::nd::array *kwds,
          const std::map<std::string, dynd::ndt::type> &tp_vars)
      {
        PyGILState_RAII pgs;

        self_type *self = self_type::make(ckb, kernreq, ckb_offset);
        self->m_proto = dynd::ndt::callable_type::make(
            dst_tp, dynd::nd::array(src_tp, nsrc));
        self->m_pyfunc = *reinterpret_cast<PyObject **>(static_data);
        Py_XINCREF(self->m_pyfunc);
        self->m_dst_arrmeta = dst_arrmeta;
        self->m_src_arrmeta.resize(nsrc);
        copy(src_arrmeta, src_arrmeta + nsrc, self->m_src_arrmeta.begin());
        self->m_ectx = *ectx;
        return ckb_offset;
      }

      static void free(dynd::ndt::callable_type::data_type *self_af)
      {
        PyObject *pyfunc = *reinterpret_cast<PyObject **>(self_af->static_data);
        if (pyfunc) {
          PyGILState_RAII pgs;
          Py_DECREF(pyfunc);
        }
      }
    };

  } // namespace pydynd::nd::functional
} // namespace pydynd::nd
} // namespace pydynd

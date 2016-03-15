#pragma once

#include <dynd/kernels/base_kernel.hpp>
#include <dynd/func/assignment.hpp>

#include "array_conversions.hpp"
#include "type_functions.hpp"
#include "types/pyobject_type.hpp"

struct apply_pyobject_kernel
    : dynd::nd::base_strided_kernel<apply_pyobject_kernel> {
  struct static_data_type {
    PyObject *func;

    static_data_type(PyObject *func) : func(func) { Py_INCREF(func); }

    //    ~static_data_type() { Py_DECREF(func); }
  };

  // Reference to the python function object
  PyObject *m_pyfunc;
  // The concrete prototype the ckernel is for
  dynd::ndt::type m_proto;
  // The arrmeta
  const char *m_dst_arrmeta;
  std::vector<const char *> m_src_arrmeta;

  apply_pyobject_kernel() : m_pyfunc(NULL) {}

  ~apply_pyobject_kernel()
  {
    if (m_pyfunc != NULL) {
      pydynd::PyGILState_RAII pgs;
      Py_DECREF(m_pyfunc);
      get_child()->destroy();
    }
  }

  void verify_postcall_consistency(PyObject *args)
  {
    intptr_t nsrc = PyTuple_GET_SIZE(args);
    // Verify that no reference to a temporary array was kept
    for (intptr_t i = 0; i != nsrc; ++i) {
      PyObject *item = PyTuple_GET_ITEM(args, i);
      if (Py_REFCNT(item) != 1 ||
          pydynd::array_to_cpp_ref(item).get()->m_use_count != 1) {
        std::stringstream ss;
        ss << "Python callback function ";
        pydynd::pyobject_ownref pyfunc_repr(PyObject_Repr(m_pyfunc));
        ss << pydynd::pystring_as_string(pyfunc_repr.get());
        ss << ", called by dynd, held a reference to parameter ";
        ss << (i + 1) << " which contained temporary memory.";
        ss << " This is disallowed.\n";
        ss << "Python wrapper ref count: " << Py_REFCNT(item) << "\n";
        pydynd::array_to_cpp_ref(item).debug_print(ss);
        // Set all the args' data pointers to NULL as a precaution
        for (i = 0; i != nsrc; ++i) {
          pydynd::array_to_cpp_ref(item).get()->data = NULL;
        }
        throw std::runtime_error(ss.str());
      }
    }
  }

  void call(dynd::nd::array *dst, const dynd::nd::array *src)
  {
    const dynd::ndt::callable_type *fpt =
        m_proto.extended<dynd::ndt::callable_type>();
    intptr_t nsrc = fpt->get_npos();

    std::vector<char *> src_data(nsrc);
    for (int i = 0; i < nsrc; ++i) {
      src_data[i] = const_cast<char *>(src[i].cdata());
    }

    single(const_cast<char *>(dst->cdata()), src_data.data());
  }

  void single(char *dst, char *const *src)
  {
    const dynd::ndt::callable_type *fpt =
        m_proto.extended<dynd::ndt::callable_type>();
    intptr_t nsrc = fpt->get_npos();
    const dynd::ndt::type &dst_tp = fpt->get_return_type();
    const std::vector<dynd::ndt::type> &src_tp = fpt->get_pos_types();
    // First set up the parameters in a tuple
    pydynd::pyobject_ownref args(PyTuple_New(nsrc));
    for (intptr_t i = 0; i != nsrc; ++i) {
      dynd::ndt::type tp = src_tp[i];
      dynd::nd::array n(
          reinterpret_cast<dynd::array_preamble *>(
              dynd::make_array_memory_block(tp.get_arrmeta_size()).get()),
          true);
      n.get()->tp = tp;
      n.get()->flags = dynd::nd::read_access_flag;
      n.get()->data = const_cast<char *>(src[i]);
      if (src_tp[i].get_arrmeta_size() > 0) {
        src_tp[i].extended()->arrmeta_copy_construct(
            n.get()->metadata(), m_src_arrmeta[i],
            dynd::intrusive_ptr<dynd::memory_block_data>());
      }
      PyTuple_SET_ITEM(args.get(), i, pydynd::array_from_cpp(std::move(n)));
    }
    // Now call the function
    pydynd::pyobject_ownref res(PyObject_Call(m_pyfunc, args.get(), NULL));
    // Copy the result into the destination memory
    PyObject *child_obj = res.get();
    char *child_src = reinterpret_cast<char *>(&child_obj);
    get_child()->single(dst, &child_src);
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
    const std::vector<dynd::ndt::type> &src_tp = fpt->get_pos_types();
    // First set up the parameters in a tuple
    pydynd::pyobject_ownref args(PyTuple_New(nsrc));
    for (intptr_t i = 0; i != nsrc; ++i) {
      dynd::ndt::type tp = src_tp[i];
      dynd::nd::array n(
          reinterpret_cast<dynd::array_preamble *>(
              dynd::make_array_memory_block(tp.get_arrmeta_size()).get()),
          true);
      n.get()->tp = tp;
      n.get()->flags = dynd::nd::read_access_flag;
      n.get()->data = const_cast<char *>(src[i]);
      if (src_tp[i].get_arrmeta_size() > 0) {
        src_tp[i].extended()->arrmeta_copy_construct(
            n.get()->metadata(), m_src_arrmeta[i],
            dynd::intrusive_ptr<dynd::memory_block_data>());
      }
      PyTuple_SET_ITEM(args.get(), i, pydynd::array_from_cpp(std::move(n)));
    }
    // Do the loop, reusing the args we created
    for (size_t j = 0; j != count; ++j) {
      // Call the function
      pydynd::pyobject_ownref res(PyObject_Call(m_pyfunc, args.get(), NULL));
      // Copy the result into the destination memory
      PyObject *child_obj = res.get();
      char *child_src = reinterpret_cast<char *>(&child_obj);
      get_child()->single(dst, &child_src);
      res.clear();
      // Validate that the call didn't hang onto the ephemeral data
      // pointers we used. This is done after the dst assignment, because
      // the function result may have contained a reference to an argument.
      verify_postcall_consistency(args.get());
      // Increment to the next one
      dst += dst_stride;
      for (intptr_t i = 0; i != nsrc; ++i) {
        const dynd::nd::array &n =
            pydynd::array_to_cpp_ref(PyTuple_GET_ITEM(args.get(), i));
        n.get()->data += src_stride[i];
      }
    }
  }

  static void instantiate(char *static_data, char *DYND_UNUSED(data),
                          dynd::nd::kernel_builder *ckb,
                          const dynd::ndt::type &dst_tp,
                          const char *dst_arrmeta, intptr_t nsrc,
                          const dynd::ndt::type *src_tp,
                          const char *const *src_arrmeta,
                          dynd::kernel_request_t kernreq, intptr_t nkwd,
                          const dynd::nd::array *kwds,
                          const std::map<std::string, dynd::ndt::type> &tp_vars)
  {
    pydynd::PyGILState_RAII pgs;

    std::vector<dynd::ndt::type> src_tp_copy(nsrc);
    for (int i = 0; i < nsrc; ++i) {
      src_tp_copy[i] = src_tp[i];
    }

    intptr_t ckb_offset = ckb->size();
    ckb->emplace_back<apply_pyobject_kernel>(kernreq);
    apply_pyobject_kernel *self =
        ckb->get_at<apply_pyobject_kernel>(ckb_offset);
    self->m_proto = dynd::ndt::callable_type::make(dst_tp, src_tp_copy);
    self->m_pyfunc = *reinterpret_cast<PyObject **>(static_data);
    Py_XINCREF(self->m_pyfunc);
    self->m_dst_arrmeta = dst_arrmeta;
    self->m_src_arrmeta.resize(nsrc);
    copy(src_arrmeta, src_arrmeta + nsrc, self->m_src_arrmeta.begin());

    dynd::ndt::type child_src_tp = dynd::ndt::make_type<pyobject_type>();
    dynd::nd::assign::get()->instantiate(
        dynd::nd::assign::get()->static_data(), nullptr, ckb, dst_tp,
        dst_arrmeta, 1, &child_src_tp, nullptr, dynd::kernel_request_single, 0,
        nullptr, tp_vars);
  }
};

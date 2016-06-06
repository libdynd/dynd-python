#pragma once

#include <dynd/assignment.hpp>
#include <dynd/kernels/base_kernel.hpp>

#include "array_conversions.hpp"
#include "type_functions.hpp"
#include "types/pyobject_type.hpp"

struct apply_pyobject_kernel : dynd::nd::base_strided_kernel<apply_pyobject_kernel> {

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
      pydynd::with_gil pgs;
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
      if (Py_REFCNT(item) != 1 || pydynd::array_to_cpp_ref(item)->get_use_count() != 1) {
        std::stringstream ss;
        ss << "Python callback function ";
        pydynd::py_ref pyfunc_repr = pydynd::capture_if_not_null(PyObject_Repr(m_pyfunc));
        ss << pydynd::pystring_as_string(pyfunc_repr.get());
        ss << ", called by dynd, held a reference to parameter ";
        ss << (i + 1) << " which contained temporary memory.";
        ss << " This is disallowed.\n";
        ss << "Python wrapper ref count: " << Py_REFCNT(item) << "\n";
        pydynd::array_to_cpp_ref(item).debug_print(ss);
        // Set all the args' data pointers to NULL as a precaution
        throw std::runtime_error(ss.str());
      }
    }
  }

  void call(dynd::nd::array *dst, const dynd::nd::array *src)
  {
    const dynd::ndt::callable_type *fpt = m_proto.extended<dynd::ndt::callable_type>();
    intptr_t nsrc = fpt->get_narg();

    std::vector<char *> src_data(nsrc);
    for (int i = 0; i < nsrc; ++i) {
      src_data[i] = const_cast<char *>(src[i].cdata());
    }

    single(const_cast<char *>(dst->cdata()), src_data.data());
  }

  void single(char *dst, char *const *src)
  {
    const dynd::ndt::callable_type *fpt = m_proto.extended<dynd::ndt::callable_type>();
    intptr_t nsrc = fpt->get_narg();
    const dynd::ndt::type &dst_tp = fpt->get_return_type();
    const std::vector<dynd::ndt::type> &src_tp = fpt->get_argument_types();
    // First set up the parameters in a tuple
    pydynd::py_ref args = pydynd::capture_if_not_null(PyTuple_New(nsrc));
    for (intptr_t i = 0; i != nsrc; ++i) {
      dynd::ndt::type tp = src_tp[i];
      dynd::nd::array n = dynd::nd::make_array(tp, const_cast<char *>(src[i]), dynd::nd::read_access_flag);
      if (src_tp[i].get_arrmeta_size() > 0) {
        src_tp[i].extended()->arrmeta_copy_construct(n.get()->metadata(), m_src_arrmeta[i], dynd::nd::memory_block());
      }
      PyTuple_SET_ITEM(args.get(), i, pydynd::array_from_cpp(std::move(n)));
    }
    // Now call the function
    PyObject *child_obj;
    char *child_src;
    { // This scope exists to limit the lifetime of py_ref res.
      pydynd::py_ref res = pydynd::capture_if_not_null(PyObject_Call(m_pyfunc, args.get(), NULL));
      // Copy the result into the destination memory
      child_obj = res.get();
      child_src = reinterpret_cast<char *>(&child_obj);
      get_child()->single(dst, &child_src);
    }
    // Validate that the call didn't hang onto the ephemeral data
    // pointers we used. This is done after the dst assignment, because
    // the function result may have contained a reference to an argument.
    verify_postcall_consistency(args.get());
  }

  void strided(char *dst, intptr_t dst_stride, char *const *src, const intptr_t *src_stride, size_t count)
  {
    const dynd::ndt::callable_type *fpt = m_proto.extended<dynd::ndt::callable_type>();
    intptr_t nsrc = fpt->get_narg();
    const dynd::ndt::type &dst_tp = fpt->get_return_type();
    const std::vector<dynd::ndt::type> &src_tp = fpt->get_argument_types();
    // First set up the parameters in a tuple
    pydynd::py_ref args = pydynd::capture_if_not_null(PyTuple_New(nsrc));
    for (intptr_t i = 0; i != nsrc; ++i) {
      dynd::ndt::type tp = src_tp[i];
      dynd::nd::array n = dynd::nd::make_array(tp, const_cast<char *>(src[i]), dynd::nd::read_access_flag);
      if (src_tp[i].get_arrmeta_size() > 0) {
        src_tp[i].extended()->arrmeta_copy_construct(n.get()->metadata(), m_src_arrmeta[i], dynd::nd::memory_block());
      }
      PyTuple_SET_ITEM(args.get(), i, pydynd::array_from_cpp(std::move(n)));
    }
    // Do the loop, reusing the args we created
    for (size_t j = 0; j != count; ++j) {
      // Call the function
      PyObject *child_obj;
      char *child_src;
      { // Scope to hold lifetime of py_ref res.
        pydynd::py_ref res = pydynd::capture_if_not_null(PyObject_Call(m_pyfunc, args.get(), NULL));
        // Copy the result into the destination memory
        child_obj = res.get();
        child_src = reinterpret_cast<char *>(&child_obj);
        get_child()->single(dst, &child_src);
      }
      // Validate that the call didn't hang onto the ephemeral data
      // pointers we used. This is done after the dst assignment, because
      // the function result may have contained a reference to an argument.
      verify_postcall_consistency(args.get());
      // Increment to the next one
      dst += dst_stride;
      for (intptr_t i = 0; i != nsrc; ++i) {
        const dynd::nd::array &n = pydynd::array_to_cpp_ref(PyTuple_GET_ITEM(args.get(), i));
        //        n->set_data(n->get_data() + src_stride[i]);
      }
    }
  }
};

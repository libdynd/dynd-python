#include "type_functions.hpp"
#include "type_deduction.hpp"
#include "type_api.h"

dynd::ndt::type &pydynd::type_to_cpp_ref(PyObject *o)
{
  if (dynd_ndt_type_to_ptr == NULL) {
    import_dynd__ndt__type();
    // Propagate any exceptions (e.g. an import error) back to Python.
    if (PyErr_Occurred()) {
      throw std::exception();
    }
  }
  return *dynd_ndt_type_to_ptr(reinterpret_cast<dynd_ndt_type_pywrapper *>(o));
}

PyTypeObject *pydynd::get_type_pytypeobject()
{
  // Check whether import has occurred by checking a different pointer.
  // For some reason the PyTypeObject in the api header is defined as
  // a macro that already dereferences the pointer.
  if (dynd_ndt_type_to_ptr == NULL) {
    import_dynd__ndt__type();
    // Propagate any exceptions (e.g. an import error) back to Python.
    if (PyErr_Occurred()) {
      throw std::exception();
    }
  }
  return &dynd_ndt_type_pywrapper_type;
}

PyObject *pydynd::type_from_cpp(const dynd::ndt::type &t)
{
  if (wrap == NULL) {
    import_dynd__ndt__type();
    // Propagate any exceptions (e.g. an import error) back to Python.
    if (PyErr_Occurred()) {
      throw std::exception();
    }
  }
  return reinterpret_cast<PyObject *>(wrap(t));
}

dynd::ndt::type pydynd::dynd_ndt_as_cpp_type(PyObject *o)
{
  if (as_cpp_type == NULL) {
    import_dynd__ndt__type();
    // Propagate any exceptions (e.g. an import error) back to Python.
    if (PyErr_Occurred()) {
      throw std::exception();
    }
  }
  return as_cpp_type(o);
}

dynd::ndt::type pydynd::dynd_ndt_cpp_type_for(PyObject *o)
{
  if (cpp_type_for == NULL) {
    import_dynd__ndt__type();
    // Propagate any exceptions (e.g.) an import error) back to Python.
    if (PyErr_Occurred()) {
      throw std::exception();
    }
  }
  return cpp_type_for(o);
}

dynd::ndt::type pydynd::ndt_type_from_pylist(PyObject *obj)
{
  // TODO: Add ability to specify access flags (e.g. immutable)
  // Do a pass through all the data to deduce its type and shape
  std::vector<intptr_t> shape;
  dynd::ndt::type tp(dynd::void_id);
  Py_ssize_t size = PyList_GET_SIZE(obj);
  shape.push_back(size);
  for (Py_ssize_t i = 0; i < size; ++i) {
    deduce_pylist_shape_and_dtype(PyList_GET_ITEM(obj, i), shape, tp, 1);
  }

  if (tp.get_id() == dynd::void_id) {
    tp = dynd::ndt::type(dynd::int32_id);
  }

  return dynd::ndt::make_type(shape.size(), shape.data(), tp);
}

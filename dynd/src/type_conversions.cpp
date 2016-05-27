#include "type_conversions.hpp"

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

#include "array_conversions.hpp"

#include "array_api.h"
#include "callable_api.h"

dynd::nd::array &pydynd::array_to_cpp_ref(PyObject *o)
{
  if (dynd_nd_array_to_ptr == NULL) {
    import_dynd__nd__array();
    // Propagate any exceptions (e.g. an import error) back to Python.
    if (PyErr_Occurred()) {
      throw std::exception();
    }
  }
  return *dynd_nd_array_to_ptr(reinterpret_cast<dynd_nd_array_pywrapper *>(o));
}

PyTypeObject *pydynd::get_array_pytypeobject()
{
  // Check whether import has occurred by checking a different pointer.
  // For some reason the PyTypeObject in the api header is defined as
  // a macro that already dereferences the pointer.
  if (dynd_nd_array_to_ptr == NULL) {
    import_dynd__nd__array();
    // Propagate any exceptions (e.g. an import error) back to Python.
    if (PyErr_Occurred()) {
      throw std::exception();
    }
  }
  return &dynd_nd_array_pywrapper_type;
}

PyObject *pydynd::array_from_cpp(const dynd::nd::array &a)
{
  if (dynd_nd_array_from_cpp == NULL) {
    import_dynd__nd__array();
    // Propagate any exceptions (e.g. an import error) back to Python.
    if (PyErr_Occurred()) {
      throw std::exception();
    }
  }
  return reinterpret_cast<PyObject *>(dynd_nd_array_from_cpp(a));
}

dynd::nd::callable &pydynd::callable_to_cpp_ref(PyObject *o)
{
  if (dynd_nd_callable_to_ptr == NULL) {
    import_dynd__nd__callable();
    // Propagate any exceptions (e.g. an import error) back to Python.
    if (PyErr_Occurred()) {
      throw std::exception();
    }
  }
  return *dynd_nd_callable_to_ptr(reinterpret_cast<dynd_nd_callable_pywrapper *>(o));
}

PyTypeObject *pydynd::get_callable_pytypeobject()
{
  // Check whether import has occurred by checking a different pointer.
  // For some reason the PyTypeObject in the api header is defined as
  // a macro that already dereferences the pointer.
  if (dynd_nd_callable_to_ptr == NULL) {
    import_dynd__nd__callable();
    // Propagate any exceptions (e.g. an import error) back to Python.
    if (PyErr_Occurred()) {
      throw std::exception();
    }
  }
  return &dynd_nd_callable_pywrapper_type;
}

PyObject *pydynd::callable_from_cpp(const dynd::nd::callable &c)
{
  if (wrap == NULL) {
    import_dynd__nd__callable();
    // Propagate any exceptions (e.g. an import error) back to Python.
    if (PyErr_Occurred()) {
      throw std::exception();
    }
  }
  return reinterpret_cast<PyObject *>(wrap(c));
}

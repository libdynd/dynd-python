#include "conversions.hpp"

#include "array_api.h"
#include "callable_api.h"
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
  // Check whether import has occured by checking a different pointer.
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
  if (dynd_ndt_type_from_cpp == NULL) {
    import_dynd__ndt__type();
    // Propagate any exceptions (e.g. an import error) back to Python.
    if (PyErr_Occurred()) {
      throw std::exception();
    }
  }
  return reinterpret_cast<PyObject *>(dynd_ndt_type_from_cpp(t));
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
  // Check whether import has occured by checking a different pointer.
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
  return *dynd_nd_callable_to_ptr(
      reinterpret_cast<dynd_nd_callable_pywrapper *>(o));
}

PyTypeObject *pydynd::get_callable_pytypeobject()
{
  // Check whether import has occured by checking a different pointer.
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
  if (dynd_nd_callable_from_cpp == NULL) {
    import_dynd__nd__callable();
    // Propagate any exceptions (e.g. an import error) back to Python.
    if (PyErr_Occurred()) {
      throw std::exception();
    }
  }
  return reinterpret_cast<PyObject *>(dynd_nd_callable_from_cpp(c));
}

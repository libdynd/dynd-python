#include "conversions.hpp"

#include "type_api.h"
#include "array_api.h"
#include "callable_api.h"

dynd::ndt::type &pydynd::type_to_cpp_ref(PyObject *o){
  if(dynd_ndt_type_to_ptr == NULL) {
    import_dynd__ndt__type();
  }
  return *dynd_ndt_type_to_ptr(reinterpret_cast<dynd_ndt_type_pywrapper*>(o));
}

PyTypeObject *pydynd::get_type_pytypeobject(){
  // Check whether import has occured by checking a different pointer.
  // For some reason the PyTypeObject in the api header is defined as
  // a macro that already dereferences the pointer.
  if(dynd_ndt_type_to_ptr == NULL) {
    import_dynd__ndt__type();
  }
  return &dynd_ndt_type_pywrapper_type;
}

PyObject *pydynd::type_from_cpp(dynd::ndt::type t){
  if(dynd_ndt_type_from_cpp == NULL) {
    import_dynd__ndt__type();
  }
  return reinterpret_cast<PyObject *>(dynd_ndt_type_from_cpp(t));
}

dynd::nd::array &pydynd::array_to_cpp_ref(PyObject *o){
  if(dynd_nd_array_to_ptr == NULL) {
    import_dynd__nd__array();
  }
  return *dynd_nd_array_to_ptr(reinterpret_cast<dynd_nd_array_pywrapper*>(o));
}

PyTypeObject *pydynd::get_array_pytypeobject(){
  // Check whether import has occured by checking a different pointer.
  // For some reason the PyTypeObject in the api header is defined as
  // a macro that already dereferences the pointer.
  if(dynd_nd_array_to_ptr == NULL) {
    import_dynd__nd__array();
  }
  return &dynd_nd_array_pywrapper_type;
}

PyObject *pydynd::array_from_cpp(dynd::nd::array a){
  if(dynd_nd_array_from_cpp == NULL) {
    import_dynd__nd__array();
  }
  return reinterpret_cast<PyObject *>(dynd_nd_array_from_cpp(a));
}

dynd::nd::callable &pydynd::callable_to_cpp_ref(PyObject *o){
  if(dynd_nd_callable_to_ptr == NULL) {
    import_dynd__nd__callable();
  }
  return *dynd_nd_callable_to_ptr(reinterpret_cast<dynd_nd_callable_pywrapper*>(o));
}

PyTypeObject *pydynd::get_callable_pytypeobject(){
  // Check whether import has occured by checking a different pointer.
  // For some reason the PyTypeObject in the api header is defined as
  // a macro that already dereferences the pointer.
  if(dynd_nd_callable_to_ptr == NULL) {
    import_dynd__nd__callable();
  }
  return &dynd_nd_callable_pywrapper_type;
}

PyObject *pydynd::callable_from_cpp(dynd::nd::callable c){
  if(dynd_nd_callable_from_cpp == NULL) {
    import_dynd__nd__callable();
  }
  return reinterpret_cast<PyObject *>(dynd_nd_callable_from_cpp(c));
}

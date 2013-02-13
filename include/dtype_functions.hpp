//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// This header defines some wrapping functions to
// access various dtype parameters
//

#ifndef _DYND__DTYPE_FUNCTIONS_HPP_
#define _DYND__DTYPE_FUNCTIONS_HPP_

#include "Python.h"

#include <sstream>

#include <dynd/dtype.hpp>
#include <dynd/string_encodings.hpp>

namespace pydynd {

/**
 * This is the typeobject and struct of w_dtype from Cython.
 */
extern PyTypeObject *WDType_Type;
inline bool WDType_CheckExact(PyObject *obj) {
    return Py_TYPE(obj) == WDType_Type;
}
inline bool WDType_Check(PyObject *obj) {
    return PyObject_TypeCheck(obj, WDType_Type);
}
struct WDType {
  PyObject_HEAD;
  // This is dtype_placement_wrapper in Cython-land
  dynd::dtype v;
};
void init_w_dtype_typeobject(PyObject *type);

inline std::string dtype_str(const dynd::dtype& d)
{
    std::stringstream ss;
    ss << d;
    return ss.str();
}

inline std::string dtype_repr(const dynd::dtype& d)
{
    std::stringstream ss;
    if (d.is_builtin() &&
                    d.get_type_id() != dynd::complex_float32_type_id &&
                    d.get_type_id() != dynd::complex_float64_type_id) {
        ss << "ndt." << d;
    } else {
        ss << "nd.dtype('" << d << "')";
    }
    return ss.str();
}

/**
 * Returns the kind of the dtype, as a Python string.
 */
PyObject *dtype_get_kind(const dynd::dtype& d);

/**
 * Returns the type id of the dtype, as a Python string.
 */
PyObject *dtype_get_type_id(const dynd::dtype& d);

/**
 * Produces a dynd::dtype corresponding to the object's type. This
 * is to determine the dtype of an object that contains a value
 * or values.
 */
dynd::dtype deduce_dtype_from_object(PyObject* obj);

/**
 * Converts a Python type, Numpy dtype, or string
 * into a dtype. This raises an error if given an object
 * which contains values, it is for dtype-like things only.
 */
dynd::dtype make_dtype_from_object(PyObject* obj);

/**
 * Constructs a w_dtype object from a dtype.
 */
PyObject * dtype_as_pyobject(const dynd::dtype& dt);

/**
 * Creates a convert dtype.
 */
dynd::dtype dnd_make_convert_dtype(const dynd::dtype& to_dtype, const dynd::dtype& from_dtype, PyObject *errmode);

/**
 * Creates a fixed-sized string dtype.
 */
dynd::dtype dnd_make_fixedstring_dtype(intptr_t size, PyObject *encoding_obj);

/**
 * Creates a blockref string dtype.
 */
dynd::dtype dnd_make_string_dtype(PyObject *encoding_obj);

/**
 * Creates a blockref pointer dtype.
 */
dynd::dtype dnd_make_pointer_dtype(const dynd::dtype& target_dtype);

dynd::dtype dnd_make_struct_dtype(PyObject *field_types, PyObject *field_names);
dynd::dtype dnd_make_fixedstruct_dtype(PyObject *field_types, PyObject *field_names);
dynd::dtype dnd_make_fixedarray_dtype(PyObject *shape, const dynd::dtype& element_dtype, PyObject *axis_perm);

/**
 * Implementation of __getitem__ for the wrapped dtype object.
 */
dynd::dtype dtype_getitem(const dynd::dtype& d, PyObject *subscript);

} // namespace pydynd

#endif // _DYND__DTYPE_FUNCTIONS_HPP_

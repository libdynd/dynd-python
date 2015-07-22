//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__ARRAY_FROM_PY_TYPEDEDUCTION_HPP_
#define _DYND__ARRAY_FROM_PY_TYPEDEDUCTION_HPP_

#include <Python.h>

#include <dynd/array.hpp>

#include "config.hpp"

namespace pydynd {

/**
 * Produces a ndt::type corresponding to the object's type. This
 * is to determine the ndt::type of an object that contains a value
 * or values.
 *
 * This function doesn't recursively process lists, or return
 * a shape if the object is a dynd or numpy array, for example.
 *
 * \param obj  The python object whose type is to be deduced.
 * \param throw_on_unknown  If true (the default), raises an exception
 *                          when no type can be deduced, otherwise returns
 *                          an uninitialized type.
 *
 * \return A type for the object, or an uninitialized type if
 *         `throw_on_unknown` is false, and nothing could be deduced.
 */
dynd::ndt::type deduce__type_from_pyobject(PyObject *obj,
                                              bool throw_on_unknown = true);

/**
 * An enumeration for describing information known about a particular
 * dimension of an array during the process of deducing its shape.
 * Dimensions start as uninitialized, and can take either nonnegative
 * values for fixed dimensions, or a value from this enum.
 */
enum shape_deduction_t {
  pydynd_shape_deduction_var = -1,
  pydynd_shape_deduction_ragged = -2,
  pydynd_shape_deduction_dict = -3,
  pydynd_shape_deduction_uninitialized = -4
};

/**
 * This function iterates over the elements of the provided
 * object, recursively deducing the shape and data type
 * of it as an array.
 *
 * \param obj  The Python object to analyze.
 * \param shape  The shape being built up. It should start as an empty
 *               vector, and gets updated and extended by this
 *               function as needed.
 * \param tp  The data type to start the deduction from. It should start as
 *            void_type_id, and is updated in place. This is updated to
 *            uninitialized_type_id if a value is encountered which can't be
 *            deduced.
 * \param current_axis  The index of the axis within the shape corresponding
 *                      to the object.
 */
void deduce_pylist_shape_and_dtype(PyObject *obj, std::vector<intptr_t> &shape,
                                   dynd::ndt::type &tp, size_t current_axis);

/**
 * This function iterates over the elements of the provided
 * object, deducing the shape of it as an array.
 *
 * This is used when constructing an array from an ndt::type which
 * has all the dimensions there, just some of them are unknown,
 * because they are strided dimensions.
 *
 * \param obj  The Python object to analyze.
 * \param ndim  The number of dimensions in `shape`.
 * \param shape  The shape being filled. It must have size `ndim`.
 */
void deduce_pyseq_shape(PyObject *obj, size_t ndim, intptr_t *shape);

/**
 * This function iterates over the elements of the provided
 * object, using the provided type as context to deduce the
 * shape of the array.
 *
 * An example where this context changes the deduction a bit
 * is for structures. To allow [(0, 1), (3, 4)] to fill a
 * two-element structure, it would be deduced as one dimensional
 * instead of two dimensional like normal.
 *
 * \param obj  The Python object to analyze.
 * \param tp  The data type providing context for the type deduction.
 * \param shape  The shape being built up. It should start as an empty vector.
 * \param initial_pass  A flag indicating whether this is the first time
 *                      visiting the current_axis.
 * \param current_axis  The index of the axis within the shape corresponding
 *                      to the object.
 */
void deduce_pyseq_shape_using_dtype(PyObject *obj, const dynd::ndt::type &tp,
                                    std::vector<intptr_t> &shape,
                                    bool initial_pass, size_t current_axis);

/**
 * Returns the number of dimensions without raggedness, treating
 * both structs and array dimensions as non-scalar.
 *
 * Examples:
 *    "3 * int32" -> 1
 *    "3 * {x: int32, y: int32}" -> 2
 *    "3 * {x: {a: int32}, y: int32}" -> 2
 *    "3 * {x: {a: int32}, y: {a: int32, b: int32}}" -> 3
 */
size_t
get_nonragged_dim_count(const dynd::ndt::type &tp,
                        size_t max_count = std::numeric_limits<size_t>::max());

/**
 * Analyzes the Python object against the dynd type to heuristically
 * determine whether copying should broadcast it as a scalar or consume
 * a dimension of the object.
 */
bool broadcast_as_scalar(const dynd::ndt::type &tp, PyObject *obj);

void init_array_from_py_typededuction();

} // namespace pydynd

#endif // _DYND__ARRAY_FROM_PY_TYPEDEDUCTION_HPP_

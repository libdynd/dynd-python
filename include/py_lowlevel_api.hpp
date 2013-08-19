//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__PY_LOWLEVEL_API_HPP_
#define _DYND__PY_LOWLEVEL_API_HPP_

#include <dynd/lowlevel_api.hpp>

#include "array_functions.hpp"
#include "type_functions.hpp"

namespace pydynd {

/**
 * This struct contains a bunch of function which provide
 * low level C-level access to the innards of dynd's python
 * exposure.
 *
 * These functions are static and should not be modified
 * after initialization.
 */
struct py_lowlevel_api_t {
    uintptr_t version;
    // Extracts the dynd object pointers from their Python wrappers.
    // These functions do not check the type of the arguments.
    dynd::array_preamble *(*get_array_ptr)(WArray *obj);
    const dynd::base_type *(*get_base_type_ptr)(WType *obj);
    PyObject *(*array_from_ptr)(PyObject *dt, PyObject *ptr, PyObject *owner, PyObject *access);
    PyObject *(*make_assignment_kernel)(PyObject *dst_dt_obj, PyObject *src_dt_obj, PyObject *kerntype, void *out_cki_ptr);
    PyObject *(*numpy_typetuples_from_ufunc)(PyObject *ufunc);
};

} // namespace pydynd

/**
 * Returns a pointer to the static low level API structure.
 */
extern "C" const void *dynd_get_py_lowlevel_api();

#endif // _DYND__PY_LOWLEVEL_API_HPP_

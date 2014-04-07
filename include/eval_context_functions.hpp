//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// This header defines some wrapping functions to
// access various nd::array parameters
//

#ifndef _DYND__EVAL_CONTEXT_FUNCTIONS_HPP_
#define _DYND__EVAL_CONTEXT_FUNCTIONS_HPP_

#include <Python.h>

#include <dynd/eval/eval_context.hpp>

namespace pydynd {

/**
 * This is the typeobject and struct of w_eval_context from Cython.
 */
extern PyTypeObject *WEvalContext_Type;
inline bool WEvalContext_CheckExact(PyObject *obj) {
    return Py_TYPE(obj) == WEvalContext_Type;
}
inline bool WEvalContext_Check(PyObject *obj) {
    return PyObject_TypeCheck(obj, WEvalContext_Type);
}
struct WEvalContext {
    PyObject_HEAD;
    const dynd::eval::eval_context *ectx;
    bool own_ectx;
};
void init_w_eval_context_typeobject(PyObject *type);

/**
 * Makes a copy of an eval context, owned by a WEvalContext.
 */
inline PyObject *wrap_eval_context(const dynd::eval::eval_context *ectx) {
    WEvalContext *result = (WEvalContext *)WEvalContext_Type->tp_alloc(WEvalContext_Type, 0);
    if (!result) {
        throw std::runtime_error("");
    }
    result->own_ectx = false;
    result->ectx = new dynd::eval::eval_context(*ectx);
    result->own_ectx = true;
    return (PyObject *)result;
}

inline const dynd::eval::eval_context *eval_context_from_pyobj(PyObject *obj)
{
    if (obj == NULL || obj == Py_None) {
        return &dynd::eval::default_eval_context;
    } else if (WEvalContext_Check(obj)) {
        return ((WEvalContext *)obj)->ectx;
    } else {
        throw std::invalid_argument(
            "invalid ectx parameter, require an nd.eval_context()");
    }
}

/**
 * Makes a copy of eval::default_eval_context, setting parameters
 * in the keyword args.
 */
dynd::eval::eval_context *new_eval_context(PyObject *kwargs);

PyObject *get_eval_context_default_errmode(PyObject *ectx_obj);
PyObject *get_eval_context_default_cuda_device_errmode(PyObject *ectx_obj);
PyObject *get_eval_context_date_parse_order(PyObject *ectx_obj);
PyObject *get_eval_context_century_window(PyObject *ectx_obj);
PyObject *get_eval_context_repr(PyObject *ectx_obj);

} // namespace pydynd

#endif // _DYND__EVAL_CONTEXT_FUNCTIONS_HPP_

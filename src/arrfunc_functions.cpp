//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "arrfunc_functions.hpp"
#include "array_functions.hpp"
#include "array_from_py.hpp"
#include "array_assign_from_py.hpp"
#include "type_functions.hpp"
#include "utility_functions.hpp"
#include "numpy_interop.hpp"

#include <dynd/types/string_type.hpp>
#include <dynd/types/base_uniform_dim_type.hpp>
#include <dynd/memblock/external_memory_block.hpp>
#include <dynd/array_range.hpp>
#include <dynd/type_promotion.hpp>
#include <dynd/types/base_struct_type.hpp>
#include <dynd/types/base_bytes_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/strided_dim_type.hpp>
#include <dynd/view.hpp>

using namespace std;
using namespace dynd;
using namespace pydynd;

PyTypeObject *pydynd::WArrFunc_Type;

void pydynd::init_w_arrfunc_typeobject(PyObject *type)
{
    WArrFunc_Type = (PyTypeObject *)type;
}

PyObject *pydynd::arrfunc_call(PyObject *af_obj, PyObject *args_obj, PyObject *ectx_obj)
{
    if (!WArrFunc_Check(af_obj)) {
        PyErr_SetString(PyExc_TypeError, "arrfunc_call expected an nd.arrfunc");
        return NULL;
    }
    const nd::arrfunc& af = ((WArrFunc *)af_obj)->v;
    if (af.is_null()) {
        PyErr_SetString(PyExc_ValueError, "cannot call a null nd.arrfunc");
        return NULL;
    }
    if (!PyTuple_Check(args_obj)) {
        PyErr_SetString(PyExc_ValueError, "arrfunc_call requires a tuple of arguments");
        return NULL;
    }
    // Convert args into nd::arrays
    intptr_t args_size = PyTuple_Size(args_obj);
    std::vector<nd::array> args(args_size);
    for (intptr_t i = 0; i < args_size; ++i) {
        args[i] = array_from_py(PyTuple_GET_ITEM(args_obj, i), 0, false);
    }
    nd::array result = af.call(args_size, args_size ? &args[0] : NULL,
                               eval_context_from_pyobj(ectx_obj));
    return wrap_array(result);
}

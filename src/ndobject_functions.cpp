//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "ndobject_functions.hpp"
#include "ndobject_from_py.hpp"
#include "dtype_functions.hpp"
#include "utility_functions.hpp"
#include "numpy_interop.hpp"

#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/memblock/external_memory_block.hpp>
#include <dynd/nodes/scalar_node.hpp>
#include <dynd/nodes/groupby_node.hpp>
#include <dynd/ndobject_arange.hpp>
#include <dynd/dtype_promotion.hpp>

using namespace std;
using namespace dynd;
using namespace pydynd;

PyTypeObject *pydynd::WNDObject_Type;

void pydynd::init_w_ndobject_typeobject(PyObject *type)
{
    WNDObject_Type = (PyTypeObject *)type;
}

dynd::ndobject pydynd::ndobject_vals(const dynd::ndobject& n)
{
    return n.vals();
}

dynd::ndobject pydynd::ndobject_eval_copy(const dynd::ndobject& n, PyObject* access_flags, const eval::eval_context *ectx)
{
    if (access_flags == Py_None) {
        return n.eval_copy(ectx);
    } else {
        return n.eval_copy(ectx, pyarg_access_flags(access_flags));
    }
}

dynd::ndobject pydynd::ndobject_cast_scalars(const dynd::ndobject& n, const dtype& dt, PyObject *assign_error_obj)
{
    return n.cast_scalars(dt, pyarg_error_mode(assign_error_obj));
}

PyObject *pydynd::ndobject_get_shape(const dynd::ndobject& n)
{
    int ndim = n.get_dtype().get_uniform_ndim();
    dimvector result(ndim);
    n.get_shape(result.get());
    return intptr_array_as_tuple(ndim, result.get());
}

PyObject *pydynd::ndobject_get_strides(const dynd::ndobject& n)
{
    int ndim = n.get_dtype().get_uniform_ndim();
    dimvector result(ndim);
    n.get_strides(result.get());
    return intptr_array_as_tuple(ndim, result.get());
}

dynd::ndobject pydynd::ndobject_getitem(const dynd::ndobject& n, PyObject *subscript)
{
    // Convert the pyobject into an array of iranges
    intptr_t size;
    shortvector<irange> indices;
    if (!PyTuple_Check(subscript)) {
        // A single subscript
        size = 1;
        indices.init(1);
        indices[0] = pyobject_as_irange(subscript);
    } else {
        size = PyTuple_GET_SIZE(subscript);
        // Tuple of subscripts
        indices.init(size);
        for (Py_ssize_t i = 0; i < size; ++i) {
            indices[i] = pyobject_as_irange(PyTuple_GET_ITEM(subscript, i));
        }
    }

    // Do an indexing operation
    return n.at_array((int)size, indices.get());
}

ndobject pydynd::ndobject_arange(PyObject *start, PyObject *stop, PyObject *step)
{
    ndobject start_nd, stop_nd, step_nd;
    if (start != Py_None) {
        ndobject_init_from_pyobject(start_nd, start);
    } else {
        start_nd = 0;
    }
    ndobject_init_from_pyobject(stop_nd, stop);
    if (step != Py_None) {
        ndobject_init_from_pyobject(step_nd, step);
    } else {
        step_nd = 1;
    }
    
    dtype dt = promote_dtypes_arithmetic(start_nd.get_dtype(),
            promote_dtypes_arithmetic(stop_nd.get_dtype(), step_nd.get_dtype()));
    
    start_nd = start_nd.cast_scalars(dt, assign_error_none).vals();
    stop_nd = stop_nd.cast_scalars(dt, assign_error_none).vals();
    step_nd = step_nd.cast_scalars(dt, assign_error_none).vals();

    if (!start_nd.is_scalar() || !stop_nd.is_scalar() || !step_nd.is_scalar()) {
        throw runtime_error("dynd::arange should only be called with scalar parameters");
    }

    return arange(dt, start_nd.get_readonly_originptr(),
            stop_nd.get_readonly_originptr(),
            step_nd.get_readonly_originptr());
}

dynd::ndobject pydynd::ndobject_linspace(PyObject *start, PyObject *stop, PyObject *count)
{
    ndobject start_nd, stop_nd;
    intptr_t count_val = pyobject_as_index(count);
    ndobject_init_from_pyobject(start_nd, start);
    ndobject_init_from_pyobject(stop_nd, stop);
    dtype dt = promote_dtypes_arithmetic(start_nd.get_dtype(), stop_nd.get_dtype());
    // Make sure it's at least floating point
    if (dt.kind() == bool_kind || dt.kind() == int_kind || dt.kind() == uint_kind) {
        dt = make_dtype<double>();
    }
    start_nd = start_nd.cast_scalars(dt, assign_error_none).vals();
    stop_nd = stop_nd.cast_scalars(dt, assign_error_none).vals();

    if (!start_nd.is_scalar() || !stop_nd.is_scalar()) {
        throw runtime_error("dynd::linspace should only be called with scalar parameters");
    }

    return linspace(dt, start_nd.get_readonly_originptr(), stop_nd.get_readonly_originptr(), count_val);
}

dynd::ndobject pydynd::ndobject_groupby(const dynd::ndobject& data, const dynd::ndobject& by, const dynd::dtype& groups)
{
    throw runtime_error("pydynd::ndobject_groupby not implemented");
//    return ndobject(make_groupby_node(data.get_node(), by.get_node(), groups));
}

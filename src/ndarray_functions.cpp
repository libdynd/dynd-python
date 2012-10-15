//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "ndarray_functions.hpp"
#include "ndarray_from_py.hpp"
#include "dtype_functions.hpp"
#include "utility_functions.hpp"
#include "numpy_interop.hpp"

#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/memblock/external_memory_block.hpp>
#include <dynd/nodes/scalar_node.hpp>
#include <dynd/nodes/groupby_node.hpp>
#include <dynd/ndarray_arange.hpp>
#include <dynd/dtype_promotion.hpp>

using namespace std;
using namespace dynd;
using namespace pydynd;

PyTypeObject *pydynd::WNDArray_Type;

void pydynd::init_w_ndarray_typeobject(PyObject *type)
{
    WNDArray_Type = (PyTypeObject *)type;
}

dynd::ndarray pydynd::ndarray_vals(const dynd::ndarray& n)
{
    return n.vals();
}

dynd::ndarray pydynd::ndarray_eval_copy(const dynd::ndarray& n, PyObject* access_flags, const eval::eval_context *ectx)
{
    if (access_flags == Py_None) {
        return n.eval_copy(ectx);
    } else {
        return n.eval_copy(ectx, pyarg_access_flags(access_flags));
    }
}

dynd::ndarray pydynd::ndarray_as_dtype(const dynd::ndarray& n, const dtype& dt, PyObject *assign_error_obj)
{
    return n.as_dtype(dt, pyarg_error_mode(assign_error_obj));
}

dynd::ndarray pydynd::ndarray_getitem(const dynd::ndarray& n, PyObject *subscript)
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
    return n.index((int)size, indices.get());
}

ndarray pydynd::ndarray_arange(PyObject *start, PyObject *stop, PyObject *step)
{
    ndarray start_nd, stop_nd, step_nd;
    if (start != Py_None) {
        ndarray_init_from_pyobject(start_nd, start);
    } else {
        start_nd = 0;
    }
    ndarray_init_from_pyobject(stop_nd, stop);
    if (step != Py_None) {
        ndarray_init_from_pyobject(step_nd, step);
    } else {
        step_nd = 1;
    }
    
    dtype dt = promote_dtypes_arithmetic(start_nd.get_dtype(),
            promote_dtypes_arithmetic(stop_nd.get_dtype(), step_nd.get_dtype()));
    
    start_nd = start_nd.as_dtype(dt, assign_error_none).vals();
    stop_nd = stop_nd.as_dtype(dt, assign_error_none).vals();
    step_nd = step_nd.as_dtype(dt, assign_error_none).vals();

    if (start_nd.get_ndim() > 0 || stop_nd.get_ndim() > 0 || step_nd.get_ndim()) {
        throw runtime_error("dynd::arange should only be called with scalar parameters");
    }

    return arange(dt, start_nd.get_readonly_originptr(),
            stop_nd.get_readonly_originptr(),
            step_nd.get_readonly_originptr());
}

dynd::ndarray pydynd::ndarray_linspace(PyObject *start, PyObject *stop, PyObject *count)
{
    ndarray start_nd, stop_nd;
    intptr_t count_val = pyobject_as_index(count);
    ndarray_init_from_pyobject(start_nd, start);
    ndarray_init_from_pyobject(stop_nd, stop);
    dtype dt = promote_dtypes_arithmetic(start_nd.get_dtype(), stop_nd.get_dtype());
    // Make sure it's at least floating point
    if (dt.kind() == bool_kind || dt.kind() == int_kind || dt.kind() == uint_kind) {
        dt = make_dtype<double>();
    }
    start_nd = start_nd.as_dtype(dt, assign_error_none).vals();
    stop_nd = stop_nd.as_dtype(dt, assign_error_none).vals();

    if (start_nd.get_ndim() > 0 || stop_nd.get_ndim() > 0) {
        throw runtime_error("dynd::linspace should only be called with scalar parameters");
    }

    return linspace(dt, start_nd.get_readonly_originptr(), stop_nd.get_readonly_originptr(), count_val);
}

dynd::ndarray pydynd::ndarray_groupby(const dynd::ndarray& data, const dynd::ndarray& by, const dynd::dtype& groups)
{
    return ndarray(make_groupby_node(data.get_node(), by.get_node(), groups));
}

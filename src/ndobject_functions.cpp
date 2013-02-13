//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "ndobject_functions.hpp"
#include "ndobject_from_py.hpp"
#include "dtype_functions.hpp"
#include "utility_functions.hpp"
#include "numpy_interop.hpp"

#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/memblock/external_memory_block.hpp>
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

dynd::ndobject pydynd::ndobject_eval(const dynd::ndobject& n)
{
    return n.vals();
}

dynd::ndobject pydynd::ndobject_eval_copy(const dynd::ndobject& n,
                PyObject* access, const eval::eval_context *ectx)
{
    uint32_t access_flags = pyarg_strings_to_int(
                    access, "access", read_access_flag|write_access_flag,
                        "readwrite", read_access_flag|write_access_flag,
                        "immutable", read_access_flag|immutable_access_flag);
    return n.eval_copy(ectx, access_flags);
}

dynd::ndobject pydynd::ndobject_empty(const dynd::dtype& d)
{
    return ndobject(d);
}

dynd::ndobject pydynd::ndobject_empty(PyObject *shape, const dynd::dtype& d)
{
    std::vector<intptr_t> shape_vec;
    pyobject_as_vector_intp(shape, shape_vec);
    return ndobject(make_ndobject_memory_block(d, (int)shape_vec.size(),
                    shape_vec.empty() ? NULL : &shape_vec[0]));
}

dynd::ndobject pydynd::ndobject_cast_scalars(const dynd::ndobject& n, const dtype& dt, PyObject *assign_error_obj)
{
    return n.cast_scalars(dt, pyarg_error_mode(assign_error_obj));
}

dynd::ndobject pydynd::ndobject_cast_udtype(const dynd::ndobject& n, const dtype& dt, PyObject *assign_error_obj)
{
    return n.cast_udtype(dt, pyarg_error_mode(assign_error_obj));
}

PyObject *pydynd::ndobject_get_shape(const dynd::ndobject& n)
{
    size_t ndim = n.get_dtype().get_undim();
    dimvector result(ndim);
    n.get_shape(result.get());
    return intptr_array_as_tuple(ndim, result.get());
}

PyObject *pydynd::ndobject_get_strides(const dynd::ndobject& n)
{
    size_t ndim = n.get_dtype().get_undim();
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
    return dynd::linspace(start_nd, stop_nd, count_val);
}

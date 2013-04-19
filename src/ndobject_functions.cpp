//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "ndobject_functions.hpp"
#include "ndobject_from_py.hpp"
#include "ndobject_assign_from_py.hpp"
#include "dtype_functions.hpp"
#include "utility_functions.hpp"
#include "numpy_interop.hpp"

#include <dynd/dtypes/string_dtype.hpp>
#include <dynd/dtypes/base_uniform_dim_dtype.hpp>
#include <dynd/memblock/external_memory_block.hpp>
#include <dynd/ndobject_arange.hpp>
#include <dynd/dtype_promotion.hpp>
#include <dynd/dtypes/base_struct_dtype.hpp>
#include <dynd/dtypes/base_bytes_dtype.hpp>
#include <dynd/dtypes/struct_dtype.hpp>

using namespace std;
using namespace dynd;
using namespace pydynd;

PyTypeObject *pydynd::WNDObject_Type;

void pydynd::init_w_ndobject_typeobject(PyObject *type)
{
    WNDObject_Type = (PyTypeObject *)type;
}

PyObject *pydynd::ndobject_str(const dynd::ndobject& n)
{
#if PY_VERSION_HEX >= 0x03000000
    // In Python 3, str is unicode
    return ndobject_unicode(n);
#else
    ndobject n_str;
    if (n.get_dtype().get_kind() == string_kind &&
                    static_cast<const base_string_dtype *>(
                        n.get_dtype().extended())->get_encoding() == string_encoding_ascii) {
        // If it's already an ASCII string, pass-through
        n_str = n;
    } else {
        // Otherwise, convert to an ASCII string
        n_str = empty(make_string_dtype(string_encoding_ascii));
        n_str.vals() = n;
    }
    const base_string_dtype *bsd =
                    static_cast<const base_string_dtype *>(n_str.get_dtype().extended());
    const char *begin = NULL, *end = NULL;
    bsd->get_string_range(&begin, &end, n_str.get_ndo_meta(), n_str.get_readonly_originptr());
    return PyString_FromStringAndSize(begin, end - begin);
#endif
}

#if PY_VERSION_HEX >= 0x03030000
#  define DYND_PY_ENCODING (string_encoding_utf_8)
#else
#  if Py_UNICODE_SIZE == 2
#    define DYND_PY_ENCODING (string_encoding_ucs_2)
#  else
#    define DYND_PY_ENCODING (string_encoding_utf_32)
#  endif
#endif

PyObject *pydynd::ndobject_unicode(const dynd::ndobject& n)
{
    ndobject n_str;
    if (n.get_dtype().get_kind() == string_kind &&
                    static_cast<const base_string_dtype *>(
                        n.get_dtype().extended())->get_encoding() == DYND_PY_ENCODING) {
        // If it's already a unicode string, pass-through
        n_str = n;
    } else {
        // Otherwise, convert to a unicode string
        n_str = empty(make_string_dtype(DYND_PY_ENCODING));
        n_str.vals() = n;
    }
    const base_string_dtype *bsd =
                    static_cast<const base_string_dtype *>(n_str.get_dtype().extended());
    const char *begin = NULL, *end = NULL;
    bsd->get_string_range(&begin, &end, n_str.get_ndo_meta(), n_str.get_readonly_originptr());
#if PY_VERSION_HEX >= 0x03030000
    // TODO: Might be more efficient to use a different Python 3 API,
    //       avoiding the creation of intermediate UTF-8
    return PyUnicode_FromStringAndSize(begin, end - begin);
#else
    return PyUnicode_FromUnicode(reinterpret_cast<const Py_UNICODE *>(begin),
                    (end - begin) / sizeof(Py_UNICODE));
#endif
}

#undef DYND_PY_ENCODING

PyObject *pydynd::ndobject_index(const dynd::ndobject& n)
{
    // Implements the nb_index slot
    switch (n.get_dtype().get_kind()) {
        case int_kind:
        case uint_kind:
            return ndobject_as_py(n);
        default:
            PyErr_SetString(PyExc_TypeError,
                            "dynd ndobject must have kind 'int'"
                            " or 'uint' to be used as an index");
            return NULL;
    }
}

PyObject *pydynd::ndobject_nonzero(const dynd::ndobject& n)
{
    // Implements the nonzero/conversion to boolean slot
    switch (n.get_dtype().value_dtype().get_kind()) {
        case bool_kind:
        case int_kind:
        case uint_kind:
        case real_kind:
        case complex_kind:
            // Follow Python in not raising errors here
            if (n.as<bool>(assign_error_none)) {
                Py_INCREF(Py_True);
                return Py_True;
            } else {
                Py_INCREF(Py_False);
                return Py_False;
            }
        case string_kind: {
            // Follow Python, return True if the string is nonempty, False otherwise
            ndobject n_eval = n.eval();
            const base_string_dtype *bsd = static_cast<const base_string_dtype *>(n_eval.get_dtype().extended());
            const char *begin = NULL, *end = NULL;
            bsd->get_string_range(&begin, &end, n_eval.get_ndo_meta(), n_eval.get_readonly_originptr());
            if (begin != end) {
                Py_INCREF(Py_True);
                return Py_True;
            } else {
                Py_INCREF(Py_False);
                return Py_False;
            }
        }
        case bytes_kind: {
            // Return True if there is a non-zero byte, False otherwise
            ndobject n_eval = n.eval();
            const base_bytes_dtype *bbd = static_cast<const base_bytes_dtype *>(n_eval.get_dtype().extended());
            const char *begin = NULL, *end = NULL;
            bbd->get_bytes_range(&begin, &end, n_eval.get_ndo_meta(), n_eval.get_readonly_originptr());
            while (begin != end) {
                if (*begin != 0) {
                    Py_INCREF(Py_True);
                    return Py_True;
                } else {
                    ++begin;
                }
            }
            Py_INCREF(Py_False);
            return Py_False;
        }
        case datetime_kind: {
            // Dates and datetimes are never zero
            // TODO: What to do with NA value?
            Py_INCREF(Py_True);
            return Py_True;
        }
        default:
            // TODO: Implement nd.any and nd.all, mention them
            //       here like NumPy does.
            PyErr_SetString(PyExc_ValueError,
                            "the truth value of a dynd array with "
                            "non-scalar type is ambiguous");
            throw exception();
    }
}

void pydynd::ndobject_init_from_pyobject(dynd::ndobject& n, PyObject* obj, PyObject *dt, bool uniform)
{
    n = ndobject_from_py(obj, make_dtype_from_pyobject(dt), uniform);
}

void pydynd::ndobject_init_from_pyobject(dynd::ndobject& n, PyObject* obj)
{
    n = ndobject_from_py(obj);
}

dynd::ndobject pydynd::ndobject_eval(const dynd::ndobject& n)
{
    return n.eval();
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
    return empty(d);
}

dynd::ndobject pydynd::ndobject_empty(PyObject *shape, const dynd::dtype& d)
{
    std::vector<intptr_t> shape_vec;
    pyobject_as_vector_intp(shape, shape_vec, true);
    return ndobject(make_ndobject_memory_block(d, (int)shape_vec.size(),
                    shape_vec.empty() ? NULL : &shape_vec[0]));
}

namespace {
    struct contains_data {
        const char *x_data;
        comparison_kernel *k;
        bool found;
    };

    void contains_callback(const dtype &DYND_UNUSED(dt), char *data,
                    const char *DYND_UNUSED(metadata), void *callback_data)
    {
        contains_data *cd = reinterpret_cast<contains_data *>(callback_data);
        if (!cd->found && (*cd->k)(cd->x_data, data)) {
            cd->found = true;
        }
    }

} // anonymous namespace

bool pydynd::ndobject_contains(const dynd::ndobject& n, PyObject *x)
{
    if (n.get_ndo() == NULL) {
        return false;
    }
    if (n.get_undim() == 0) {
        // TODO: Allow for struct types, etc?
        throw runtime_error("cannot call __contains__ on a scalar ndobject");
    }

    // Turn 'n' into dtype/metadata/data with a uniform_dim leading dimension
    ndobject tmp;
    dtype dt;
    const base_uniform_dim_dtype *budd;
    const char *metadata, *data;
    if (n.get_dtype().get_kind() == uniform_dim_kind) {
        dt = n.get_dtype();
        budd = static_cast<const base_uniform_dim_dtype *>(dt.extended());
        metadata = n.get_ndo_meta();
        data = n.get_readonly_originptr();
    } else {
        tmp = n.eval();
        if (tmp.get_dtype().get_kind() != uniform_dim_kind) {
            throw runtime_error("internal error in ndobject_contains: expected uniform_dim kind after eval() call");
        }
        dt = tmp.get_dtype();
        budd = static_cast<const base_uniform_dim_dtype *>(dt.extended());
        metadata = tmp.get_ndo_meta();
        data = tmp.get_readonly_originptr();
    }

    // Turn 'x' into an ndobject, and make a comparison kernel
    ndobject x_ndo = ndobject_from_py(x);
    const dtype& x_dt = x_ndo.get_dtype();
    const char *x_metadata = x_ndo.get_ndo_meta();
    const char *x_data = x_ndo.get_readonly_originptr();
    const dtype& child_dt = budd->get_element_dtype();
    const char *child_metadata = metadata + budd->get_element_metadata_offset();
    comparison_kernel k;
    try {
        make_comparison_kernel(&k, 0,
                    x_dt, x_metadata, child_dt, child_metadata,
                    comparison_type_equal, &eval::default_eval_context);
    } catch(const not_comparable_error&) {
        return false;
    }

    contains_data aux;
    aux.x_data = x_data;
    aux.k = &k;
    aux.found = false;
    budd->foreach_leading(const_cast<char *>(data), metadata, &contains_callback, &aux);
    return aux.found;
}

dynd::ndobject pydynd::ndobject_ucast(const dynd::ndobject& n, const dtype& dt,
                size_t replace_undim, PyObject *assign_error_obj)
{
    return n.ucast(dt, replace_undim, pyarg_error_mode(assign_error_obj));
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

static void pyobject_as_irange_array(intptr_t& out_size, shortvector<irange>& out_indices,
                PyObject *subscript)
{
    if (!PyTuple_Check(subscript)) {
        // A single subscript
        out_size = 1;
        out_indices.init(1);
        out_indices[0] = pyobject_as_irange(subscript);
    } else {
        out_size = PyTuple_GET_SIZE(subscript);
        // Tuple of subscripts
        out_indices.init(out_size);
        for (Py_ssize_t i = 0; i < out_size; ++i) {
            out_indices[i] = pyobject_as_irange(PyTuple_GET_ITEM(subscript, i));
        }
    }
}

dynd::ndobject pydynd::ndobject_getitem(const dynd::ndobject& n, PyObject *subscript)
{
    if (subscript == Py_Ellipsis) {
        return n.at_array(0, NULL);
    } else {
        // Convert the pyobject into an array of iranges
        intptr_t size;
        shortvector<irange> indices;
        pyobject_as_irange_array(size, indices, subscript);

        // Do an indexing operation
        return n.at_array(size, indices.get());
    }
}

void pydynd::ndobject_setitem(const dynd::ndobject& n, PyObject *subscript, PyObject *value)
{
    // TODO: Write a mechanism for assigning directly
    //       from PyObject to ndobject
    if (subscript == Py_Ellipsis) {
        ndobject_broadcast_assign_from_py(n, value);
#if PY_VERSION_HEX < 0x03000000
    } else if (PyInt_Check(subscript)) {
        long i = PyInt_AS_LONG(subscript);
        const char *metadata = n.get_ndo_meta();
        char *data = n.get_readwrite_originptr();
        dtype d = n.get_dtype().at_single(i, &metadata, const_cast<const char **>(&data));
        ndobject_broadcast_assign_from_py(d, metadata, data, value);
#endif // PY_VERSION_HEX < 0x03000000
    } else if (PyLong_Check(subscript)) {
        intptr_t i = PyLong_AsSsize_t(subscript);
        if (i == -1 && PyErr_Occurred()) {
            throw runtime_error("error converting int value");
        }
        const char *metadata = n.get_ndo_meta();
        char *data = n.get_readwrite_originptr();
        dtype d = n.get_dtype().at_single(i, &metadata, const_cast<const char **>(&data));
        ndobject_broadcast_assign_from_py(d, metadata, data, value);
    } else {
        intptr_t size;
        shortvector<irange> indices;
        pyobject_as_irange_array(size, indices, subscript);
        ndobject_broadcast_assign_from_py(n.at_array(size, indices.get(), false), value);
    }
}

ndobject pydynd::ndobject_arange(PyObject *start, PyObject *stop, PyObject *step, PyObject *dt)
{
    ndobject start_nd, stop_nd, step_nd;
    dtype dt_nd;

    if (start != Py_None) {
        start_nd = ndobject_from_py(start);
    } else {
        start_nd = 0;
    }
    stop_nd = ndobject_from_py(stop);
    if (step != Py_None) {
        step_nd = ndobject_from_py(step);
    } else {
        step_nd = 1;
    }

    if (dt != Py_None) {
        dt_nd = make_dtype_from_pyobject(dt);
    } else {
        dt_nd = promote_dtypes_arithmetic(start_nd.get_dtype(),
                    promote_dtypes_arithmetic(stop_nd.get_dtype(), step_nd.get_dtype()));
    }
    
    start_nd = start_nd.ucast(dt_nd).eval();
    stop_nd = stop_nd.ucast(dt_nd).eval();
    step_nd = step_nd.ucast(dt_nd).eval();

    if (!start_nd.is_scalar() || !stop_nd.is_scalar() || !step_nd.is_scalar()) {
        throw runtime_error("dynd::arange should only be called with scalar parameters");
    }

    return arange(dt_nd, start_nd.get_readonly_originptr(),
            stop_nd.get_readonly_originptr(),
            step_nd.get_readonly_originptr());
}

dynd::ndobject pydynd::ndobject_linspace(PyObject *start, PyObject *stop, PyObject *count, PyObject *dt)
{
    ndobject start_nd, stop_nd;
    intptr_t count_val = pyobject_as_index(count);
    start_nd = ndobject_from_py(start);
    stop_nd = ndobject_from_py(stop);
    if (dt == Py_None) {
        return dynd::linspace(start_nd, stop_nd, count_val);
    } else {
        return dynd::linspace(start_nd, stop_nd, count_val, make_dtype_from_pyobject(dt));
    }
}

dynd::ndobject pydynd::nd_fields(const ndobject& n, PyObject *field_list)
{
    vector<string> selected_fields;
    pyobject_as_vector_string(field_list, selected_fields);

    // TODO: Move this implementation into dynd
    dtype fdt = n.get_udtype();
    if (fdt.get_kind() != struct_kind) {
        stringstream ss;
        ss << "nd.fields must be given an ndobject of 'struct' kind, not ";
        ss << fdt;
        throw runtime_error(ss.str());
    }
    const base_struct_dtype *bsd = static_cast<const base_struct_dtype *>(fdt.extended());
    const dtype *field_types = bsd->get_field_types();

    if (selected_fields.empty()) {
        throw runtime_error("nd.fields requires at least one field name to be specified");
    }
    // Construct the field mapping and output field dtypes
    vector<intptr_t> selected_index(selected_fields.size());
    vector<dtype> selected_dtypes(selected_fields.size());
    for (size_t i = 0; i != selected_fields.size(); ++i) {
        selected_index[i] = bsd->get_field_index(selected_fields[i]);
        if (selected_index[i] < 0) {
            stringstream ss;
            ss << "field name ";
            print_escaped_utf8_string(ss, selected_fields[i]);
            ss << " does not exist in dtype " << fdt;
            throw runtime_error(ss.str());
        }
        selected_dtypes[i] = field_types[selected_index[i]];
    }
    // Create the result udt
    dtype rudt = make_struct_dtype(selected_dtypes, selected_fields);
    dtype rdt = n.get_dtype().with_replaced_udtype(rudt);
    const base_struct_dtype *rudt_bsd = static_cast<const base_struct_dtype *>(rudt.extended());

    // Allocate the new memory block.
    size_t metadata_size = rdt.get_metadata_size();
    ndobject result(make_ndobject_memory_block(metadata_size));

    // Clone the data pointer
    result.get_ndo()->m_data_pointer = n.get_ndo()->m_data_pointer;
    result.get_ndo()->m_data_reference = n.get_ndo()->m_data_reference;
    if (result.get_ndo()->m_data_reference == NULL) {
        result.get_ndo()->m_data_reference = n.get_memblock().get();
    }
    memory_block_incref(result.get_ndo()->m_data_reference);

    // Copy the flags
    result.get_ndo()->m_flags = n.get_ndo()->m_flags;

    // Set the dtype and transform the metadata
    result.get_ndo()->m_dtype = dtype(rdt).release();
    // First copy all the uniform dtype metadata
    dtype tmp_dt = rdt;
    char *dst_metadata = result.get_ndo_meta();
    const char *src_metadata = n.get_ndo_meta();
    while (tmp_dt.get_undim() > 0) {
        if (tmp_dt.get_kind() != uniform_dim_kind) {
            throw runtime_error("nd.fields doesn't support dimensions with pointers yet");
        }
        const base_uniform_dim_dtype *budd = static_cast<const base_uniform_dim_dtype *>(
                        tmp_dt.extended());
        size_t offset = budd->metadata_copy_construct_onedim(dst_metadata, src_metadata,
                        n.get_memblock().get());
        dst_metadata += offset;
        src_metadata += offset;
        tmp_dt = budd->get_element_dtype();
    }
    // Then create the metadata for the new struct
    const size_t *metadata_offsets = bsd->get_metadata_offsets();
    const size_t *result_metadata_offsets = rudt_bsd->get_metadata_offsets();
    const size_t *data_offsets = bsd->get_data_offsets(src_metadata);
    size_t *result_data_offsets = reinterpret_cast<size_t *>(dst_metadata);
    for (size_t i = 0; i != selected_fields.size(); ++i) {
        const dtype& dt = selected_dtypes[i];
        // Copy the data offset
        result_data_offsets[i] = data_offsets[selected_index[i]];
        // Copy the metadata for this field
        if (dt.get_metadata_size() > 0) {
            dt.extended()->metadata_copy_construct(dst_metadata + result_metadata_offsets[i],
                            src_metadata + metadata_offsets[selected_index[i]],
                            n.get_memblock().get());
        }
    }

    return result;
}

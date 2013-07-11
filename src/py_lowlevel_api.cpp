//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <sstream>

#include <dynd/kernels/assignment_kernels.hpp>
#include<dynd/memblock/external_memory_block.hpp>

#include "py_lowlevel_api.hpp"
#include "utility_functions.hpp"
#include "exception_translation.hpp"

using namespace std;
using namespace dynd;
using namespace pydynd;

namespace {
    dynd::array_preamble *get_array_ptr(WArray *obj)
    {
        return obj->v.get_ndo();
    }

    const dynd::base_type *get_base_type_ptr(WType *obj)
    {
        return obj->v.extended();
    }

    PyObject *array_from_ptr(PyObject *dt, PyObject *ptr, PyObject *owner, PyObject *access)
    {
        try {
            ndt::type d = make_ndt_type_from_pyobject(dt);
            size_t ptr_val = pyobject_as_size_t(ptr);
            uint32_t access_flags = pyarg_strings_to_int(
                            access, "access", nd::read_access_flag,
                                "readwrite", nd::read_access_flag|nd::write_access_flag,
                                "readonly", nd::read_access_flag,
                                "immutable", nd::read_access_flag|nd::immutable_access_flag);
            if (d.get_metadata_size() != 0) {
                stringstream ss;
                ss << "Cannot create a dynd array from a raw pointer with non-empty metadata, type: ";
                ss << d;
                throw runtime_error(ss.str());
            }
            nd::array result(make_array_memory_block(0));
            d.swap(result.get_ndo()->m_type);
            result.get_ndo()->m_data_pointer = reinterpret_cast<char *>(ptr_val);
            memory_block_ptr owner_memblock = make_external_memory_block(owner, &py_decref_function);
            Py_INCREF(owner);
            result.get_ndo()->m_data_reference = owner_memblock.release();
            result.get_ndo()->m_flags = access_flags;
            return wrap_array(DYND_MOVE(result));
        } catch(...) {
            translate_exception();
            return NULL;
        }
    }

    PyObject *make_assignment_kernel(PyObject *dst_dt_obj, PyObject *src_dt_obj, PyObject *kerntype_obj, void *out_dki_ptr)
    {
        try {
            dynamic_kernel_instance *out_dki = reinterpret_cast<dynamic_kernel_instance *>(out_dki_ptr);
            out_dki->kernel = NULL;
            out_dki->kernel_size = 0;
            out_dki->free_func = NULL;

            ndt::type dst_dt = make_ndt_type_from_pyobject(dst_dt_obj);
            ndt::type src_dt = make_ndt_type_from_pyobject(src_dt_obj);
            if (dst_dt.get_metadata_size() != 0) {
                stringstream ss;
                ss << "Cannot create an assignment kernel independent of metadata with non-empty metadata, type: ";
                ss << dst_dt;
                throw runtime_error(ss.str());
            }
            if (src_dt.get_metadata_size() != 0) {
                stringstream ss;
                ss << "Cannot create an assignment kernel independent of metadata with non-empty metadata, type: ";
                ss << src_dt;
                throw runtime_error(ss.str());
            }
            string kt = pystring_as_string(kerntype_obj);
            kernel_request_t kerntype;
            if (kt == "single") {
                kerntype = kernel_request_single;
            } else if (kt == "strided") {
                kerntype = kernel_request_strided;
            } else {
                stringstream ss;
                ss << "Invalid kernel request type ";
                print_escaped_utf8_string(ss, kt);
                throw runtime_error(ss.str());
            }

            hierarchical_kernel hk;
            size_t kernel_size = make_assignment_kernel(&hk, 0, dst_dt, NULL,
                            src_dt, NULL, kerntype, assign_error_default,
                            &eval::default_eval_context);
            hk.move_into_dki(out_dki, kernel_size);

            Py_INCREF(Py_None);
            return Py_None;
        } catch(...) {
            translate_exception();
            return NULL;
        }
    }

    const py_lowlevel_api_t py_lowlevel_api = {
        0, // version, should increment this every time the struct changes at a release
        &get_array_ptr,
        &get_base_type_ptr,
        &array_from_ptr,
        &make_assignment_kernel,
    };
} // anonymous namespace

extern "C" const void *dynd_get_py_lowlevel_api()
{
    return reinterpret_cast<const void *>(&py_lowlevel_api);
}

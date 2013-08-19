//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <sstream>

#include <dynd/kernels/assignment_kernels.hpp>
#include<dynd/memblock/external_memory_block.hpp>

#include "py_lowlevel_api.hpp"
#include "numpy_ufunc_kernel.hpp"
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

    PyObject *make_assignment_kernel(PyObject *dst_tp_obj, PyObject *src_tp_obj, PyObject *kerntype_obj, void *out_cki_ptr)
    {
        try {
            ckernel_instance *out_cki = reinterpret_cast<ckernel_instance *>(out_cki_ptr);
            out_cki->kernel = NULL;
            out_cki->kernel_size = 0;
            out_cki->free_func = NULL;

            ndt::type dst_tp = make_ndt_type_from_pyobject(dst_tp_obj);
            ndt::type src_tp = make_ndt_type_from_pyobject(src_tp_obj);
            if (dst_tp.get_metadata_size() != 0) {
                stringstream ss;
                ss << "Cannot create an assignment kernel independent of metadata with non-empty metadata, type: ";
                ss << dst_tp;
                throw runtime_error(ss.str());
            }
            if (src_tp.get_metadata_size() != 0) {
                stringstream ss;
                ss << "Cannot create an assignment kernel independent of metadata with non-empty metadata, type: ";
                ss << src_tp;
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

            ckernel_builder ckb;
            size_t kernel_size = make_assignment_kernel(&ckb, 0, dst_tp, NULL,
                            src_tp, NULL, kerntype, assign_error_default,
                            &eval::default_eval_context);
            ckb.move_into_cki(out_cki, kernel_size);

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
        &pydynd::numpy_typetuples_from_ufunc
    };
} // anonymous namespace

extern "C" const void *dynd_get_py_lowlevel_api()
{
    return reinterpret_cast<const void *>(&py_lowlevel_api);
}

//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <sstream>

#include <dynd/kernels/assignment_kernels.hpp>

#include "py_lowlevel_api.hpp"
#include "utility_functions.hpp"
#include "exception_translation.hpp"

using namespace std;
using namespace dynd;
using namespace pydynd;

namespace {
    dynd::ndobject_preamble *get_ndobject_ptr(WNDObject *obj)
    {
        return obj->v.get_ndo();
    }

    const dynd::base_dtype *get_base_dtype_ptr(WDType *obj)
    {
        return obj->v.extended();
    }

    PyObject *make_assignment_kernel(PyObject *dst_dt_obj, PyObject *src_dt_obj, PyObject *kerntype_obj, void *out_dki_ptr)
    {
        try {
            dynamic_kernel_instance *out_dki = reinterpret_cast<dynamic_kernel_instance *>(out_dki_ptr);
            out_dki->kernel = NULL;
            out_dki->kernel_size = 0;
            out_dki->free_func = NULL;

            dtype dst_dt = make_dtype_from_pyobject(dst_dt_obj);
            dtype src_dt = make_dtype_from_pyobject(src_dt_obj);
            if (dst_dt.get_metadata_size() != 0) {
                stringstream ss;
                ss << "Cannot create an assignment kernel independent of metadata with non-empty metadata, dtype: ";
                ss << dst_dt;
                throw runtime_error(ss.str());
            }
            if (src_dt.get_metadata_size() != 0) {
                stringstream ss;
                ss << "Cannot create an assignment kernel independent of metadata with non-empty metadata, dtype: ";
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
        0, // version, should increment this everytime the struct changes
        &get_ndobject_ptr,
        &get_base_dtype_ptr,
        &make_assignment_kernel,
    };
} // anonymous namespace

extern "C" const void *dynd_get_py_lowlevel_api()
{
    return reinterpret_cast<const void *>(&py_lowlevel_api);
}

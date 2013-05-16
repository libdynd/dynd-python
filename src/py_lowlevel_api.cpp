//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "py_lowlevel_api.hpp"

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

    const py_lowlevel_api_t py_lowlevel_api = {
        0, // version, should increment this everytime the struct changes
        &get_ndobject_ptr,
        &get_base_dtype_ptr,
    };
} // anonymous namespace

extern "C" const void *dynd_get_py_lowlevel_api()
{
    return reinterpret_cast<const void *>(&py_lowlevel_api);
}

//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__ELWISE_GFUNC_FUNCTIONS_HPP_
#define _DND__ELWISE_GFUNC_FUNCTIONS_HPP_

#include <Python.h>

#include <sstream>
#include <deque>
#include <vector>

#include <dnd/dtype.hpp>
#include <dnd/ndarray.hpp>
#include <dnd/kernels/kernel_instance.hpp>
#include <dnd/codegen/codegen_cache.hpp>
#include <dnd/gfunc/elwise_gfunc.hpp>

namespace pydynd {

void elwise_gfunc_add_kernel(dynd::gfunc::elwise& gf, dynd::codegen_cache& cgcache, PyObject *kernel);

PyObject *elwise_gfunc_call(dynd::gfunc::elwise& gf, PyObject *args, PyObject *kwargs);

inline std::string elwise_gfunc_debug_dump(dynd::gfunc::elwise& gf)
{
    std::stringstream ss;
    gf.debug_dump(ss);
    return ss.str();
}

struct elwise_gfunc_placement_wrapper {
    intptr_t dummy[(sizeof(dynd::gfunc::elwise) + sizeof(intptr_t) - 1)/sizeof(intptr_t)];
};

inline void elwise_gfunc_placement_new(elwise_gfunc_placement_wrapper& v, const char *name)
{
    // Call placement new
    new (&v) dynd::gfunc::elwise(name);
}

inline void elwise_gfunc_placement_delete(elwise_gfunc_placement_wrapper& v)
{
    // Call the destructor
    ((dynd::gfunc::elwise *)(&v))->~elwise();
}

// placement cast
inline dynd::gfunc::elwise& GET(elwise_gfunc_placement_wrapper& v)
{
    return *(dynd::gfunc::elwise *)&v;
}

} // namespace pydynd

#endif // _DND__ELWISE_GFUNC_FUNCTIONS_HPP_

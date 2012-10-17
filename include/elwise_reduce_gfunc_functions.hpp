//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__ELWISE_REDUCE_GFUNC_FUNCTIONS_HPP_
#define _DND__ELWISE_REDUCE_GFUNC_FUNCTIONS_HPP_

#include <Python.h>

#include <sstream>
#include <deque>
#include <vector>

#include <dynd/dtype.hpp>
#include <dynd/ndarray.hpp>
#include <dynd/kernels/kernel_instance.hpp>
#include <dynd/codegen/codegen_cache.hpp>
#include <dynd/gfunc/elwise_reduce_gfunc.hpp>

namespace pydynd {

void elwise_reduce_gfunc_add_kernel(dynd::gfunc::elwise_reduce& gf, dynd::codegen_cache& cgcache, PyObject *kernel,
                            bool associative, bool commutative, const dynd::ndarray& identity);

PyObject *elwise_reduce_gfunc_call(dynd::gfunc::elwise_reduce& gf, PyObject *args, PyObject *kwargs);

inline std::string elwise_reduce_gfunc_debug_dump(dynd::gfunc::elwise_reduce& gf)
{
    std::stringstream ss;
    gf.debug_dump(ss);
    return ss.str();
}

struct elwise_reduce_gfunc_placement_wrapper {
    intptr_t dummy[(sizeof(dynd::gfunc::elwise_reduce) + sizeof(intptr_t) - 1)/sizeof(intptr_t)];
};

inline void placement_new(elwise_reduce_gfunc_placement_wrapper& v, const char *name)
{
    // Call placement new
    new (&v) dynd::gfunc::elwise_reduce(name);
}

inline void placement_delete(elwise_reduce_gfunc_placement_wrapper& v)
{
    // Call the destructor
    ((dynd::gfunc::elwise_reduce *)(&v))->~elwise_reduce();
}

// placement cast
inline dynd::gfunc::elwise_reduce& GET(elwise_reduce_gfunc_placement_wrapper& v)
{
    return *(dynd::gfunc::elwise_reduce *)&v;
}

} // namespace pydynd

#endif // _DND__ELWISE_REDUCE_GFUNC_FUNCTIONS_HPP_

//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__ELWISE_REDUCE_GFUNC_FUNCTIONS_HPP_
#define _DYND__ELWISE_REDUCE_GFUNC_FUNCTIONS_HPP_

#include <Python.h>

#include <sstream>
#include <deque>
#include <vector>

#include <dynd/type.hpp>
#include <dynd/array.hpp>
#include <dynd/codegen/codegen_cache.hpp>
#include <dynd/gfunc/elwise_reduce_gfunc.hpp>

namespace pydynd {

void elwise_reduce_gfunc_add_kernel(dynd::gfunc::elwise_reduce& gf, dynd::codegen_cache& cgcache, PyObject *kernel,
                            bool associative, bool commutative, const dynd::nd::array& identity);

PyObject *elwise_reduce_gfunc_call(dynd::gfunc::elwise_reduce& gf, PyObject *args, PyObject *kwargs);

inline std::string elwise_reduce_gfunc_debug_print(dynd::gfunc::elwise_reduce& gf)
{
    std::stringstream ss;
    //gf.debug_print(ss); TODO reenable
    ss << "temporarily disabled\n";
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

#endif // _DYND__ELWISE_REDUCE_GFUNC_FUNCTIONS_HPP_

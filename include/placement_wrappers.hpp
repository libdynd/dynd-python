//
// Copyright (C) 2011-13 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//
// This header defines some placement wrappers of ndt::type and nd::array
// to enable wrapping them without adding extra indirection layers.
//

#ifndef _DYND__PLACEMENT_WRAPPERS_HPP_
#define _DYND__PLACEMENT_WRAPPERS_HPP_

#include <dynd/type.hpp>
#include <dynd/array.hpp>
//#include <dynd/codegen/codegen_cache.hpp>
#include <dynd/vm/elwise_program.hpp>
#include "gfunc_callable_functions.hpp"

#define DYND_DEFINE_PLACEMENT_WRAPPER(cpp_type, cpp_type_no_namespace, wrapper_type) \
    /** This is a struct with the same alignment (because of intptr_t) \
     // and size as ndt::type. It's what we wrap in Cython, and use \
     // placement new and delete to manage its lifetime. */ \
    struct wrapper_type { \
        intptr_t dummy[(sizeof(cpp_type) + sizeof(intptr_t) - 1)/sizeof(intptr_t)]; \
    }; \
    \
    inline void placement_new(wrapper_type& v) \
    { \
        /* Call placement new */ \
        new (&v) cpp_type(); \
    } \
    \
    inline void placement_delete(wrapper_type& v) \
    { \
        /* Call the destructor */ \
        ((cpp_type *)(&v))->~cpp_type_no_namespace(); \
    } \
    \
    /* placement cast */ \
    inline cpp_type& GET(wrapper_type& v) \
    { \
        return *(cpp_type *)&v; \
    } \
    \
    /* placement assignment */ \
    inline void SET(wrapper_type& v, const cpp_type& d) \
    { \
        *(cpp_type *)&v = d; \
    }

namespace pydynd {

DYND_DEFINE_PLACEMENT_WRAPPER(dynd::ndt::type, type, ndt_type_placement_wrapper);
DYND_DEFINE_PLACEMENT_WRAPPER(dynd::nd::array, array, array_placement_wrapper);
//DYND_DEFINE_PLACEMENT_WRAPPER(dynd::codegen_cache, codegen_cache, codegen_cache_placement_wrapper);
DYND_DEFINE_PLACEMENT_WRAPPER(dynd::vm::elwise_program, elwise_program, vm_elwise_program_placement_wrapper);
DYND_DEFINE_PLACEMENT_WRAPPER(pydynd::array_callable_wrapper, array_callable_wrapper,
            array_callable_placement_wrapper);
DYND_DEFINE_PLACEMENT_WRAPPER(pydynd::ndt_type_callable_wrapper, ndt_type_callable_wrapper,
            ndt_type_callable_placement_wrapper);

} // namespace pydynd

#endif // _DYND__PLACEMENT_WRAPPERS_HPP_

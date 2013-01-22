//
// Copyright (C) 2011-13, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DYND__CODEGEN_CACHE_FUNCTIONS_HPP_
#define _DYND__CODEGEN_CACHE_FUNCTIONS_HPP_

#include "Python.h"

#include <sstream>
#include <stdexcept>

#include <dynd/codegen/codegen_cache.hpp>

namespace pydynd {

inline std::string codegen_cache_debug_print(const dynd::codegen_cache& cgcache)
{
    std::stringstream ss;
    cgcache.debug_print(ss);
    return ss.str();
}


} // namespace pydynd

#endif // _DYND__CODEGEN_CACHE_FUNCTIONS_HPP_

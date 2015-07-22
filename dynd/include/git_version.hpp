//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _PYDYND__GIT_VERSION_HPP_
#define _PYDYND__GIT_VERSION_HPP_

#include "config.hpp"

namespace pydynd {
// These are defined in git_version.cpp, generated from
// git_version.cpp.in by the CMake build configuration.
PYDYND_API extern const char dynd_python_git_sha1[];
PYDYND_API extern const char dynd_python_version_string[];
} // namespace pydynd

#endif // _PYDYND__GIT_VERSION_HPP_

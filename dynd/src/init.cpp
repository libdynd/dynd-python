//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#define NUMPY_IMPORT_ARRAY
#include "numpy_interop_defines.hpp"

#include "assign.hpp"
#include "init.hpp"

void pydynd::numpy_interop_init() { import_numpy(); }

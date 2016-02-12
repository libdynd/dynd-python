//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#define NUMPY_IMPORT_ARRAY
#include "numpy_interop.hpp"

#include "assign.hpp"
#include "copy_from_numpy_arrfunc.hpp"
#include "init.hpp"
#include "type_functions.hpp"

void pydynd::pydynd_init()
{
  import_numpy();
  pydynd::init_type_functions();
}

//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#define NUMPY_IMPORT_ARRAY
#include "numpy_interop.hpp"

#include "init.hpp"
#include "ctypes_interop.hpp"
#include "copy_to_numpy_arrfunc.hpp"
#include "copy_from_numpy_arrfunc.hpp"
#include "copy_to_pyobject_arrfunc.hpp"
#include "copy_from_pyobject_arrfunc.hpp"
#include "array_from_py.hpp"
#include "array_from_py_dynamic.hpp"
#include "array_from_py_typededuction.hpp"
#include "type_functions.hpp"

static void pydynd_cleanup() { dynd::libdynd_cleanup(); }

void pydynd::pydynd_init()
{
  import_numpy();
  dynd::libdynd_init();
  atexit(pydynd_cleanup);
  pydynd::init_type_functions();
  pydynd::init_array_from_py_typededuction();
  pydynd::init_array_from_py_dynamic();
  pydynd::init_array_from_py();
  pydynd::init_ctypes_interop();
}

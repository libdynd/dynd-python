//
// Copyright (C) 2011-14 Mark Wiebe, DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>

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

static void pydynd_cleanup()
{
  pydynd::cleanup_copy_from_pyobject();
  pydynd::cleanup_copy_to_pyobject();
  pydynd::cleanup_copy_from_numpy();
  pydynd::cleanup_copy_to_numpy();
  dynd::libdynd_cleanup();
}

void pydynd::pydynd_init()
{
  dynd::libdynd_init();
  atexit(pydynd_cleanup);
  pydynd::init_type_functions();
  pydynd::init_array_from_py_typededuction();
  pydynd::init_array_from_py_dynamic();
  pydynd::init_array_from_py();
  pydynd::init_ctypes_interop();
  pydynd::init_copy_to_numpy();
  pydynd::init_copy_from_numpy();
  pydynd::init_copy_to_pyobject();
  pydynd::init_copy_from_pyobject();
}

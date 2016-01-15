//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "arrfunc_functions.hpp"
#include "array_functions.hpp"
#include "array_from_py.hpp"
#include "type_functions.hpp"
#include "utility_functions.hpp"
#include "numpy_interop.hpp"

#include <dynd/types/string_type.hpp>
#include <dynd/types/base_dim_type.hpp>
#include <dynd/memblock/external_memory_block.hpp>
#include <dynd/array_range.hpp>
#include <dynd/type_promotion.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/base_bytes_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/func/rolling.hpp>
#include <dynd/view.hpp>
#include <dynd/callable.hpp>
#include <dynd/func/callable_registry.hpp>

using namespace std;
using namespace dynd;
using namespace pydynd;

PyObject *pydynd::get_published_callables()
{
  pyobject_ownref res(PyDict_New());
  const map<std::string, dynd::nd::callable> &reg = func::get_regfunctions();
  for (map<std::string, dynd::nd::callable>::const_iterator it = reg.begin();
       it != reg.end(); ++it) {
    PyDict_SetItem(res.get(), pystring_from_string(it->first),
                   DyND_PyWrapper_New(it->second));
  }
  return res.release();
}

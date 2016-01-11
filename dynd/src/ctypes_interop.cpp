//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>

#include <dynd/types/fixed_string_type.hpp>
#include <dynd/types/struct_type.hpp>
#include <dynd/types/pointer_type.hpp>
#include <dynd/types/type_alignment.hpp>

#include "ctypes_interop.hpp"
#include "type_functions.hpp"
#include "utility_functions.hpp"

using namespace std;
using namespace dynd;
using namespace pydynd;

//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include "wrapper.hpp"

using namespace std;

template <typename T>
PyTypeObject *&DyND_PyWrapper_Type()
{
  static PyTypeObject *type = NULL;
  return type;
}

template PYDYND_API PyTypeObject *&DyND_PyWrapper_Type<dynd::nd::array>();
template PYDYND_API PyTypeObject *&DyND_PyWrapper_Type<dynd::nd::callable>();
template PYDYND_API PyTypeObject *&DyND_PyWrapper_Type<dynd::ndt::type>();

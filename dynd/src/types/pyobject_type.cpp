//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#include <Python.h>

#include <dynd/type_registry.hpp>

#include "types/pyobject_type.hpp"

using namespace dynd;

const type_id_t pyobject_id = new_id("pyobject", any_kind_id);

pyobject_type::pyobject_type(type_id_t new_id)
    : ndt::base_type(new_id, pyobject_id, sizeof(PyObject *), alignof(PyObject *), type_flag_none | type_flag_zeroinit,
                     0, 0, 0)
{
}

void pyobject_type::print_type(std::ostream &o) const { o << "pyobject"; }

bool pyobject_type::operator==(const base_type &rhs) const { return get_id() == rhs.get_id(); }

void pyobject_type::print_data(std::ostream &o, const char *arrmeta, const char *data) const
{
  PyObject *repr = PyObject_Repr(*reinterpret_cast<PyObject *const *>(data));
#if PY_VERSION_HEX < 0x03000000
  o << PyString_AsString(repr);
#else
  o << PyUnicode_AsUTF8(repr);
#endif
  Py_DECREF(repr);
}

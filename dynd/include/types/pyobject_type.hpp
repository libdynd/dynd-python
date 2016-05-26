//
// Copyright (C) 2011-15 DyND Developers
// BSD 2-Clause License, see LICENSE.txt
//

#pragma once

#include "visibility.hpp"

#include <dynd/types/base_type.hpp>

extern const dynd::type_id_t pyobject_id;

class PYDYND_API pyobject_type : public dynd::ndt::base_type {
public:
  pyobject_type(dynd::type_id_t new_id);

  void print_type(std::ostream &o) const;
  void print_data(std::ostream &o, const char *arrmeta, const char *data) const;

  bool operator==(const base_type &rhs) const;
};

namespace dynd {
namespace ndt {

  template <>
  struct traits<PyObject *> {
    static type equivalent() { return make_type<pyobject_type>(); }
  };

} // namespace dynd::ndt
} // namespace dynd

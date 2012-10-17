//
// Copyright (C) 2011-12, Dynamic NDArray Developers
// BSD 2-Clause License, see LICENSE.txt
//

#ifndef _DND__VM_ELWISE_PROGRAM_FUNCTIONS_HPP_
#define _DND__VM_ELWISE_PROGRAM_FUNCTIONS_HPP_

#include "Python.h"

#include <sstream>
#include <stdexcept>

#include <dynd/vm/elwise_program.hpp>

#include "placement_wrappers.hpp"

namespace pydynd {

/**
 * Converts a Python object into a VM elwise_program.
 */
void vm_elwise_program_from_py(PyObject *obj, dynd::vm::elwise_program& out_ep);

/**
 * Converts a VM elementwise program into a Python object
 */
PyObject *vm_elwise_program_as_py(dynd::vm::elwise_program& ep);

inline std::string vm_elwise_program_debug_dump(const dynd::vm::elwise_program& n)
{
    std::stringstream ss;
    n.debug_dump(ss);
    return ss.str();
}

} // namespace pydynd

#endif // _DND__VM_ELWISE_PROGRAM_FUNCTIONS_HPP_
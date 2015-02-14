#
# Copyright (C) 2011-15 DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

from simple cimport translate_exception

from libcpp.string cimport string

cdef extern from "dynd/vm/elwise_program.hpp" namespace "dynd::vm":
    cdef cppclass elwise_program:
        pass

cdef extern from "vm_elwise_program_functions.hpp" namespace "pydynd":
    void vm_elwise_program_from_py(object, elwise_program&) except +translate_exception
    object vm_elwise_program_as_py(elwise_program&) except +translate_exception
    string vm_elwise_program_debug_print(elwise_program&) except +translate_exception

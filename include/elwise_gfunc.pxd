#
# Copyright (C) 2011-13, DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

cdef extern from "<dynd/gfunc/elwise_gfunc.hpp>" namespace "dynd::gfunc":
    cdef cppclass elwise_gfunc:
        string& get_name()

cdef extern from "elwise_gfunc_functions.hpp" namespace "pydynd":
#    void elwise_gfunc_add_kernel(elwise_gfunc&, codegen_cache&, object) except +translate_exception
    object elwise_gfunc_call(elwise_gfunc&, object, object) except +translate_exception
    string elwise_gfunc_repr(elwise_gfunc&) except +translate_exception
    string elwise_gfunc_debug_print(elwise_gfunc&) except +translate_exception

    cdef struct elwise_gfunc_placement_wrapper:
        pass
    void placement_new(elwise_gfunc_placement_wrapper&, char *)
    void placement_delete(elwise_gfunc_placement_wrapper&)
    # ndarray placement cast
    elwise_gfunc& GET(elwise_gfunc_placement_wrapper&)

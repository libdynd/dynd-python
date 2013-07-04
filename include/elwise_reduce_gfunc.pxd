#
# Copyright (C) 2011-13 Mark Wiebe, DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

cdef extern from "<dynd/gfunc/elwise_reduce_gfunc.hpp>" namespace "dynd::gfunc":
    cdef cppclass elwise_reduce_gfunc:
        string& get_name()

cdef extern from "elwise_reduce_gfunc_functions.hpp" namespace "pydynd":
#    void elwise_reduce_gfunc_add_kernel(elwise_reduce_gfunc&, codegen_cache&, object,
#                            bint, bint, ndarray&) except +translate_exception
    object elwise_reduce_gfunc_call(elwise_reduce_gfunc&, object, object) except +translate_exception
    string elwise_reduce_gfunc_repr(elwise_reduce_gfunc&) except +translate_exception
    string elwise_reduce_gfunc_debug_print(elwise_reduce_gfunc&) except +translate_exception

    cdef struct elwise_reduce_gfunc_placement_wrapper:
        pass
    void placement_new(elwise_reduce_gfunc_placement_wrapper&, char *)
    void placement_delete(elwise_reduce_gfunc_placement_wrapper&)
    # nd::array placement cast
    elwise_reduce_gfunc& GET(elwise_reduce_gfunc_placement_wrapper&)

#
# Copyright (C) 2011-13, DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

cdef extern from "gfunc_callable_functions.hpp" namespace "pydynd":
    void add_dtype_names_to_dir_dict(dtype&, object) except +translate_exception
    object get_dtype_dynamic_property(dtype&, object) except +translate_exception

    void add_ndobject_names_to_dir_dict(ndobject&, object) except +translate_exception
    object get_ndobject_dynamic_property(ndobject&, object) except +translate_exception
    void set_ndobject_dynamic_property(ndobject&, object, object) except +translate_exception

    # Function property of dtype
    cdef cppclass dtype_callable_wrapper:
        pass
    object dtype_callable_call(dtype_callable_wrapper&, object, object) except +translate_exception
    object call_dtype_constructor_function(dtype&, object, object) except +translate_exception

    void init_w_dtype_callable_typeobject(object)
    cdef struct dtype_callable_placement_wrapper:
        pass
    void placement_new(dtype_callable_placement_wrapper&)
    void placement_delete(dtype_callable_placement_wrapper&)
    # ndarray placement cast
    dtype_callable_wrapper& GET(dtype_callable_placement_wrapper&)

    # Function property of ndobject
    cdef cppclass ndobject_callable_wrapper:
        pass
    object ndobject_callable_call(ndobject_callable_wrapper&, object, object) except +translate_exception

    void init_w_ndobject_callable_typeobject(object)
    cdef struct ndobject_callable_placement_wrapper:
        pass
    void placement_new(ndobject_callable_placement_wrapper&)
    void placement_delete(ndobject_callable_placement_wrapper&)
    # ndarray placement cast
    ndobject_callable_wrapper& GET(ndobject_callable_placement_wrapper&)

#
# Copyright (C) 2011-13 Mark Wiebe, DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

cdef extern from "gfunc_callable_functions.hpp" namespace "pydynd":
    void add_dtype_names_to_dir_dict(dtype&, object) except +translate_exception
    object get_dtype_dynamic_property(dtype&, object) except +translate_exception

    void add_array_names_to_dir_dict(ndarray&, object) except +translate_exception
    object get_array_dynamic_property(ndarray&, object) except +translate_exception
    void set_array_dynamic_property(ndarray&, object, object) except +translate_exception

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

    # Function property of nd::array
    cdef cppclass array_callable_wrapper:
        pass
    object array_callable_call(array_callable_wrapper&, object, object) except +translate_exception

    void init_w_array_callable_typeobject(object)
    cdef struct array_callable_placement_wrapper:
        pass
    void placement_new(array_callable_placement_wrapper&)
    void placement_delete(array_callable_placement_wrapper&)
    # ndarray placement cast
    array_callable_wrapper& GET(array_callable_placement_wrapper&)

#
# Copyright (C) 2011-12, Dynamic NDArray Developers
# BSD 2-Clause License, see LICENSE.txt
#

cdef extern from "gfunc_callable_functions.hpp" namespace "pydynd":
    void add_dtype_names_to_dir_dict(dtype&, object) except +translate_exception
    object get_dtype_dynamic_property(dtype&, object) except +translate_exception

    void add_ndobject_names_to_dir_dict(ndobject&, object) except +translate_exception
    object get_ndobject_dynamic_property(ndobject&, object) except +translate_exception

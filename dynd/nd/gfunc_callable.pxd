#
# Copyright (C) 2011-15 DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

from translate_except cimport translate_exception
from ndt.type cimport _type
from nd.array cimport _array

cdef extern from "gfunc_callable_functions.hpp" namespace "pydynd":
    void add__type_names_to_dir_dict(_type&, object) except +translate_exception
    object get__type_dynamic_property(_type&, object) except +translate_exception

    void add_array_names_to_dir_dict(_array&, object) except +translate_exception
    object get_array_dynamic_property(_array&, object) except +translate_exception
    void set_array_dynamic_property(_array&, object, object) except +translate_exception

    # Function properties
    cdef cppclass _type_callable_wrapper:
        pass
    object _type_callable_call(_type_callable_wrapper&, object, object) except +translate_exception
    object call__type_constructor_function(_type&, object, object) except +translate_exception

    void init_w__type_callable_typeobject(object)
    cdef struct _type_callable_placement_wrapper:
        pass

    # Function property of nd::array
    cdef cppclass array_callable_wrapper:
        pass
    object array_callable_call(array_callable_wrapper&, object, object) except +translate_exception

    void init_w_array_callable_typeobject(object)
    cdef struct array_callable_placement_wrapper:
        pass

cdef extern from "dynd/func/arrfunc.hpp" namespace "dynd":
    cdef cppclass ndarrfunc "dynd::nd::arrfunc":
        ndarrfunc() except +translate_exception

cdef extern from "arrfunc_from_pyfunc.hpp" namespace "pydynd::nd::functional":
    ndarrfunc apply(object, object) except +translate_exception

cdef extern from "arrfunc_functions.hpp" namespace "pydynd":
    void init_w_arrfunc_typeobject(object)
    object arrfunc_call(object, object, object, object) except +translate_exception
    object arrfunc_rolling_apply(object, object, object, object) except +translate_exception
    object dynd_get_published_arrfuncs "pydynd::get_published_arrfuncs" () except +translate_exception

    object wrap_array(const ndarrfunc &af)
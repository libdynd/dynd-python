from libc.stdint cimport intptr_t
from libcpp.string cimport string

from ..config cimport translate_exception
from ..cpp.array cimport array as _array
from ..cpp.type cimport as _type

cdef extern from "numpy_interop.hpp" namespace "pydynd":
    object numpy_dtype_obj_from_type(_type&) except +translate_exception

cdef extern from 'gfunc_callable_functions.hpp' namespace 'pydynd':
    void add__type_names_to_dir_dict(_type&, object) except +translate_exception
    object get__type_dynamic_property(_type&, object) except +translate_exception

    # Function properties
    cdef cppclass _type_callable_wrapper:
        pass
    object _type_callable_call(_type_callable_wrapper&, object, object) except +translate_exception

    void init_w__type_callable_typeobject(object)

cdef extern from 'type_functions.hpp' namespace 'pydynd':
    void init_w_type_typeobject(object)

    _type make__type_from_pyobject(object) except +translate_exception

    object _type_get_shape(_type&) except +translate_exception
    object _type_get_kind(_type&) except +translate_exception
    object _type_get_type_id(_type&) except +translate_exception
    string _type_str(_type &)
    string _type_repr(_type &)

    _type dynd_make_convert_type(_type&, _type&) except +translate_exception
    _type dynd_make_view_type(_type&, _type&) except +translate_exception
    _type dynd_make_fixed_string_type(int, object) except +translate_exception
    _type dynd_make_string_type(object) except +translate_exception
    _type dynd_make_pointer_type(_type&) except +translate_exception
    _type dynd_make_struct_type(object, object) except +translate_exception
    _type dynd_make_cstruct_type(object, object) except +translate_exception
    _type dynd_make_fixed_dim_type(object, _type&) except +translate_exception
    _type dynd_make_cfixed_dim_type(object, _type&, object) except +translate_exception

cdef class type(object):
    cdef _type v

cdef as_numba_type(_type)
cdef _type from_numba_type(object)

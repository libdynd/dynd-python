# cython: c_string_type=str, c_string_encoding=ascii

from cpython.object cimport Py_EQ, Py_NE, PyObject
from libc.stdint cimport (intptr_t, int8_t, int16_t, int32_t, int64_t,
                          uint8_t, uint16_t, uint32_t, uint64_t)
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool as cpp_bool

import datetime

from ..cpp.types.type_id cimport (type_id_t, uninitialized_id,
                                  bool_id, int8_id, int16_id,
                                  int32_id, int64_id, int128_id,
                                  uint8_id, uint16_id, uint32_id,
                                  uint64_id, uint128_id,
                                  float16_id, float32_id,
                                  float64_id, float128_id,
                                  complex_float32_id,
                                  complex_float64_id, void_id,
                                  callable_id, string_id, bytes_id,
                                  date_id, datetime_id, time_id,
                                  type_id)
from ..cpp.types.datashape_formatter cimport format_datashape as dynd_format_datashape
# from ..cpp.types.categorical_type cimport dynd_make_categorical_type
from ..cpp.types.fixed_bytes_type cimport make_fixed_bytes as dynd_make_fixed_bytes_type
from ..cpp.types.base_fixed_dim_type cimport dynd_make_fixed_dim_kind_type
from ..cpp.types.tuple_type cimport tuple_type
from ..cpp.types.struct_type cimport struct_type
from ..cpp.types.var_dim_type cimport var_dim_type as _var_dim_type
from ..cpp.types.callable_type cimport callable_type as _callable_type
from ..cpp.types.string_type cimport string_type
from ..cpp.types.bytes_type cimport make as make_bytes_type
from ..cpp.type cimport make_type
from ..cpp.complex cimport complex as dynd_complex

from ..config cimport translate_exception

import datetime as _datetime
import numpy as _np
import ctypes as _ct

cdef extern from *:
    # Declare these here rather than using the declarations in the Cython
    # wrappers for the Python headers since their versions incorrectly pass
    # a reference rather than a pointer.
    bint PyUnicode_Check(PyObject *o)
    bint PyType_Check(PyObject *o)

cdef extern from "numpy_interop.hpp":
    ctypedef struct PyArray_Descr

cdef extern from "numpy_interop.hpp" namespace "pydynd":
    _type _type_from_numpy_dtype(PyArray_Descr*)

cdef extern from "type_conversions.hpp" namespace 'pydynd':
    _type ndt_type_from_pylist(object) except +translate_exception

cdef extern from 'type_functions.hpp' namespace 'pydynd':
    object _type_get_shape(_type&) except +translate_exception
    object _type_get_id(_type&) except +translate_exception
    string _type_str(_type &)

    _type dynd_make_fixed_string_type(int, object) except +translate_exception
    _type dynd_make_string_type() except +translate_exception
    _type dynd_make_pointer_type(_type&) except +translate_exception
    _type dynd_make_struct_type(object, object) except +translate_exception
    _type dynd_make_cstruct_type(object, object) except +translate_exception
    _type dynd_make_fixed_dim_type(object, _type&) except +translate_exception
    _type dynd_make_cfixed_dim_type(object, _type&, object) except +translate_exception
    void init_type_functions()

cdef extern from "type_unpack.hpp":
    object from_type_property(const pair[_type, const char *] &) except +translate_exception


builtin_tuple = tuple

init_type_functions()
_builtin_type = __builtins__.type
_builtin_bool = __builtins__.bool

__all__ = ['type_ids', 'type', 'bool', 'int8', 'int16', 'int32', 'int64', 'int128', \
    'uint8', 'uint16', 'uint32', 'uint64', 'uint128', 'float16', 'float32', \
    'float64', 'float128', 'complex_float32', 'complex_float64', 'void', \
    'tuple', 'struct', 'callable', 'scalar', 'astype']

type_ids = {}
type_ids['UNINITIALIZED'] = uninitialized_id
type_ids['BOOL'] = bool_id
type_ids['INT8'] = int8_id
type_ids['INT16'] = int16_id
type_ids['INT32'] = int32_id
type_ids['INT64'] = int64_id
type_ids['INT128'] = int128_id
type_ids['UINT8'] = uint8_id
type_ids['UINT16'] = uint16_id
type_ids['UINT32'] = uint32_id
type_ids['UINT64'] = uint64_id
type_ids['UINT128'] = uint128_id
type_ids['FLOAT16'] = float16_id
type_ids['FLOAT32'] = float32_id
type_ids['FLOAT64'] = float64_id
type_ids['FLOAT128'] = float128_id
type_ids['COMPLEX64'] = complex_float32_id
type_ids['COMPLEX128'] = complex_float64_id
type_ids['VOID'] = void_id
type_ids['CALLABLE'] = callable_id

cdef class type(object):
    """
    ndt.type(obj=None)

    Create a dynd type object.

    A dynd type object describes the dimensional
    structure and element type of a dynd array.

    Parameters
    ----------
    obj : string or other data type, optional
        A Blaze datashape string or a data type from another
        system such as NumPy.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> ndt.type('int16')
    ndt.int16
    >>> ndt.type('5 * var * float32')
    ndt.type("5 * var * float32")
    >>> ndt.type('{x: float32, y: float32, z: float32}')
    ndt.type("{x : float32, y : float32, z : float32}")
    """

    def __init__(self, rep=None):
        if rep is not None:
            self.v = as_cpp_type(rep)

    property shape:
        """
        tp.shape
        The shape of the array dimensions of the type. For
        dimensions whose size is unknown without additional
        array arrmeta or array data, a -1 is returned.
        """
        def __get__(self):
            return _type_get_shape(self.v)

    property canonical_type:
        """
        tp.canonical_type
        Returns a version of this type that is canonical,
        where any intermediate pointers are removed and expressions
        are stripped away.
        """
        def __get__(self):
            cdef type result = type()
            result.v = self.v.get_canonical_type()
            return result

    property dshape:
        """
        tp.dshape
        The blaze datashape of the dynd type, as a string.
        """
        def __get__(self):
            return str(<char *>dynd_format_datashape(self.v).c_str())

    property data_size:
        """
        tp.data_size
        The size, in bytes, of the data for an instance
        of this dynd type.
        None is returned if array arrmeta is required to
        fully specify it. For example, both the fixed and
        struct types require such arrmeta.
        """
        def __get__(self):
            cdef ssize_t result = self.v.get_data_size()
            if result > 0:
                return result
            else:
                return None

    property default_data_size:
        """
        tp.default_data_size
        The size, in bytes, of the data for a default-constructed
        instance of this dynd type.
        """
        def __get__(self):
            return self.v.get_default_data_size()

    property data_alignment:
        """
        tp.data_alignment
        The alignment, in bytes, of the data for an
        instance of this dynd type.
        Data for this dynd type must always be aligned
        according to this alignment, unaligned data
        requires an adapter transformation applied.
        """
        def __get__(self):
            return self.v.get_data_alignment()

    property arrmeta_size:
        """
        tp.arrmeta_size
        The size, in bytes, of the arrmeta for
        this dynd type.
        """
        def __get__(self):
            return self.v.get_arrmeta_size()

    property base_id:
        def __get__(type self):
            return dynd_ndt_type_to_cpp(self).get_base_id()

    property id:
        def __get__(self):
            return self.v.get_id()

    def __getattr__(self, name):
        if self.v.is_null():
            raise AttributeError(name)

        cdef pair[_type, const char *] p = self.v.get_properties()[name]
        if p.first.is_null():
            raise AttributeError(name)

        try:
            return from_type_property(p)
        except RuntimeError:
            raise AttributeError(name)

    def __str__(self):
        return str(<char *>_type_str(self.v).c_str())

    def __repr__(self):
        return "ndt.type(" + repr(str(self)) + ")"

    def __richcmp__(lhs, rhs, int op):
        if op == Py_EQ:
            if isinstance(lhs, type) and isinstance(rhs, type):
                return (<type>lhs).v == (<type>rhs).v
            else:
                return False
        elif op == Py_NE:
            if isinstance(lhs, type) and isinstance(rhs, type):
                return (<type>lhs).v != (<type>rhs).v
            else:
                return False
        return NotImplemented

    def as_numpy(self):
        """
        tp.as_numpy()
        If possible, converts the ndt.type object into an
        equivalent numpy dtype.
        Examples
        --------
        >>> from dynd import nd, ndt
        >>> ndt.int32.as_numpy()
        dtype('int32')
        """
        return _numpy_dtype_from__type(self.v)

    def match(self, rhs):
        """
        tp.match(candidate)
        Returns True if the candidate type ``candidate`` matches the possibly
        symbolic pattern type ``tp``, False otherwise.
        Examples
        --------
        >>> from dynd import nd, ndt
        >>> ndt.type("T").match(ndt.int32)
        True
        >>> ndt.type("Dim * T").match(ndt.int32)
        False
        >>> ndt.type("M * {x: ?T, y: T}").match("10 * {x : ?int32, y : int32}")
        True
        >>> ndt.type("M * {x: ?T, y: T}").match("10 * {x : ?int32, y : ?int32}")
        False
        """
        return self.v.match(type(rhs).v)

cdef _type dynd_ndt_type_to_cpp(type t) except *:
    # Once this becomes a method of the type wrapper class, this check and
    # its corresponding exception handler declaration are no longer necessary
    # since the self parameter is guaranteed to never be None.
    if t is None:
        raise TypeError("Cannot extract DyND C++ type from None.")
    return t.v

cdef _type *dynd_ndt_type_to_ptr(type t) except *:
    # Once this becomes a method of the type wrapper class, this check and
    # its corresponding exception handler declaration are no longer necessary
    # since the self parameter is guaranteed to never be None.
    if t is None:
        raise TypeError("Cannot extract DyND C++ type from None.")
    return &(t.v)

# returns a Python object, so no exception specifier is needed.
cdef type dynd_ndt_type_from_cpp(const _type &t):
    cdef type tp = type.__new__(type)
    tp.v = t
    return tp

# TODO: The translation from Python object to C++ type is complex enough
# that we really ought to have some sort of
# Pyobject* to ndt::type mapping set up. That's not trivial to do, so
# an army of if statements will have to suffice for now.
# A dictionary-like interface would probably make more sense though.

cdef _type cpp_type_from_numpy_type(object o) except *:
    if o is _np.bool_:
        return make_type[cpp_bool]()
    elif o is _np.int8:
        return make_type[int8_t]()
    elif o is _np.int16:
        return make_type[int16_t]()
    elif o is _np.int32:
        return make_type[int32_t]()
    elif o is _np.int64:
        return make_type[int64_t]()
    elif o is _np.uint8:
        return make_type[uint8_t]()
    elif o is _np.uint16:
        return make_type[uint16_t]()
    elif o is _np.uint32:
        return make_type[uint32_t]()
    elif o is _np.uint64:
        return make_type[uint64_t]()
    elif o is _np.float32:
        return make_type[float]()
    elif o is _np.float64:
        return make_type[double]()
    elif o is _np.complex64:
        return make_type[dynd_complex[float]]()
    elif o is _np.complex128:
        return make_type[dynd_complex[double]]()

cdef _type cpp_type_from_typeobject(object o) except *:
    if o is type or o is _builtin_type:
        return make_type[_type]()
    if o is _builtin_bool:
        return make_type[cpp_bool]()
    elif o is int or o is long:
        # Once we support arbitrary precision integers,
        # Python longs should map to those instead.
        return make_type[int32_t]()
    elif o is float:
        return make_type[double]()
    elif o is complex:
        return make_type[dynd_complex[double]]()
    elif o is str or o is unicode:
        return make_type[string_type]()
    elif o is bytes:
        return make_bytes_type()
    elif o is bytearray:
        return make_bytes_type()
    elif issubclass(o, _np.generic):
        return cpp_type_from_numpy_type(o)
    raise ValueError("Cannot make ndt.type from {}.".format(o))

_ctypes_base_type = _ct.c_int.__bases__[0].__bases__[0]

cdef _type as_cpp_type(object o) except *:
    if _builtin_type(o) is type:
        return dynd_ndt_type_to_cpp(<type>o)
    elif _builtin_type(o) is str or PyUnicode_Check(<PyObject*>o):
        # Use Cython's automatic conversion to c++ strings.
        return _type(<string>o)
    elif _builtin_type(o) is int or _builtin_type(o) is long:
        return _type(<type_id_t>(<int>o))
    elif _is_numpy_dtype(<PyObject*>o):
        return _type_from_numpy_dtype(<PyArray_Descr*>o)
    elif issubclass(o, _ctypes_base_type):
        raise ValueError("Conversion from ctypes type to DyND type not currently supported.")
    elif PyType_Check(<PyObject*>o):
        return cpp_type_from_typeobject(o)
    raise ValueError("Cannot make ndt.type from {}.".format(o))

cpdef type astype(object o):
    if _builtin_type(o) is type:
        return o
    return dynd_ndt_type_from_cpp(as_cpp_type(o))

# Disabled until dynd_make_categorical_type takes a std::vector<T>
'''
def make_categorical(values):
    """
    ndt.make_categorical(values)
    Constructs a categorical dynd type with the
    specified values as its categories.
    Instances of the resulting type are integers, consisting
    of indices into this values array. The size of the values
    array controls what kind of integers are used, if there
    are 256 or fewer categories, a uint8 is used, if 65536 or
    fewer, a uint16 is used, and otherwise a uint32 is used.
    Parameters
    ----------
    values : one-dimensional array
        This is an array of the values that become the categories
        of the resulting type. The values must be unique.
    Examples
    --------
    >>> from dynd import nd, ndt
    >>> ndt.make_categorical(['sunny', 'rainy', 'cloudy', 'stormy'])
    ndt.type("categorical[string, [\"sunny\", \"rainy\", \"cloudy\", \"stormy\"]]")
    """
    cdef type result = type()
    result.v = dynd_make_categorical_type(array(values).v)
    return result
'''

def make_fixed_bytes(intptr_t data_size, intptr_t data_alignment=1):
    """
    ndt.make_fixed_bytes(data_size, data_alignment=1)
    Constructs a bytes type with the specified data size and alignment.
    Parameters
    ----------
    data_size : int
        The number of bytes of one instance of this dynd type.
    data_alignment : int, optional
        The required alignment of instances of this dynd type.
        This value must be a small power of two. Default: 1.
    Examples
    --------
    >>> from dynd import nd, ndt
    >>> ndt.make_fixed_bytes(4)
    ndt.type("bytes[4]")
    >>> ndt.make_fixed_bytes(6, 2)
    ndt.type("bytes[6, align=2]")
    """
    cdef type result = type()
    result.v = dynd_make_fixed_bytes_type(data_size, data_alignment)
    return result

def make_fixed_dim(shape, element_tp):
    """
    ndt.make_fixed_dim(shape, element_tp)
    Constructs a fixed_dim type of the given shape.
    Parameters
    ----------
    shape : tuple of int
        The multi-dimensional shape of the resulting fixed array type.
    element_tp : dynd type
        The type of each element in the resulting array type.
    Examples
    --------
    >>> from dynd import nd, ndt
    >>> ndt.make_fixed_dim(5, ndt.int32)
    ndt.type("5 * int32")
    >>> ndt.make_fixed_dim((3,5), ndt.int32)
    ndt.type("3 * 5 * int32")
    """
    cdef type result = type()
    result.v = dynd_make_fixed_dim_type(shape, type(element_tp).v)
    return result

def make_fixed_string(int size, encoding=None):
    """
    ndt.make_fixed_string(size, encoding='utf_8')
    Constructs a fixed-size string type with a specified encoding,
    whose size is the specified number of base units for the encoding.
    Parameters
    ----------
    size : int
        The number of base encoding units in the data. For example,
        for UTF-8, a size of 10 means 10 bytes. For UTF-16, a size
        of 10 means 20 bytes.
    encoding : string, optional
        The encoding used for storing unicode code points. Supported
        values are 'ascii', 'utf_8', 'utf_16', 'utf_32', 'ucs_2'.
        Default: 'utf_8'.
    Examples
    --------
    >>> from dynd import nd, ndt
    >>> ndt.make_fixed_string(10)
    ndt.type("string[10]")
    >>> ndt.make_fixed_string(10, 'utf_32')
    ndt.type("string[10,'utf32']")
    """
    cdef type result = type()
    result.v = dynd_make_fixed_string_type(size, encoding)
    return result

def make_string():
    """
    ndt.make_string()
    Constructs a variable-sized string dynd type with uft-8 encoding.
    Examples
    --------
    >>> from dynd import nd, ndt
    >>> ndt.make_string()
    ndt.string
    """
    cdef type result = type()
    result.v = dynd_make_string_type()
    return result

def make_struct(field_types, field_names):
    """
    ndt.make_struct(field_types, field_names)
    Constructs a struct dynd type, which has fields with a flexible
    per-array layout.
    If a subset of fields from a fixed_struct are taken,
    the result is a struct, with the layout specified
    in the dynd array's arrmeta.
    Parameters
    ----------
    field_types : list of dynd types
        A list of types, one for each field.
    field_names : list of strings
        A list of names, one for each field, corresponding to 'field_types'.
    Examples
    --------
    >>> from dynd import nd, ndt
    >>> ndt.make_struct([ndt.int32, ndt.float64], ['x', 'y'])
    ndt.type("{x : int32, y : float64}")
    """
    cdef type result = type()
    result.v = dynd_make_struct_type(field_types, field_names)
    return result

def make_fixed_dim_kind(element_tp, ndim=None):
    """
    ndt.make_fixed_dim_kind(element_tp, ndim=1)
    Constructs an array dynd type with one or more symbolic fixed
    dimensions. A single fixed_dim_kind dynd type corresponds
    to one dimension, so when ndim > 1, multiple fixed_dim_kind
    dimensions are created.
    Parameters
    ----------
    element_tp : dynd type
        The type of one element in the symbolic array.
    ndim : int
        The number of fixed_dim_kind dimensions to create.
    Examples
    --------
    >>> from dynd import nd, ndt
    >>> ndt.make_fixed_dim_kind(ndt.int32)
    ndt.type("Fixed * int32")
    >>> ndt.make_fixed_dim_kind(ndt.int32, 3)
    ndt.type("Fixed * Fixed * Fixed * int32")
    """
    cdef type result = type()
    if (ndim is None):
        result.v = dynd_make_fixed_dim_kind_type(type(element_tp).v)
    else:
        result.v = dynd_make_fixed_dim_kind_type(type(element_tp).v, int(ndim))
    return result


def make_var_dim(element_tp):
    """
    ndt.make_fixed_dim(element_tp)
    Constructs a var_dim type.
    Parameters
    ----------
    element_tp : dynd type
        The type of each element in the resulting array type.
    Examples
    --------
    >>> from dynd import nd, ndt
    >>> ndt.make_var_dim(ndt.float32)
    ndt.type("var * float32")
    """
    cdef type result = type()
    result.v = make_type[_var_dim_type](as_cpp_type(element_tp))
    return result

bool = type(bool_id)
int8 = type(int8_id)
int16 = type(int16_id)
int32 = type(int32_id)
int64 = type(int64_id)
int128 = type(int128_id)
uint8 = type(uint8_id)
uint16 = type(uint16_id)
uint32 = type(uint32_id)
uint64 = type(uint64_id)
uint128 = type(uint128_id)
float16 = type(float16_id)
float32 = type(float32_id)
float64 = type(float64_id)
float128 = type(float128_id)
complex_float32 = type(complex_float32_id)
complex_float64 = type(complex_float64_id)
void = type(void_id)

def tuple(*args):
    cdef vector[_type] _args

    if args:
        # TODO: Use a different interface that doesn't involve nd.array.
        # This should basicaly just go through and call as_cpp_type
        # on each argument and make a vector or array of C++ types
        # that can then be used to create a tuple type.
        # Constructing an array from the list is really
        # only partially correct.
        for arg in args:
            _args.push_back(as_cpp_type(arg))

        return dynd_ndt_type_from_cpp(make_type[tuple_type](_args.size(), _args.data()))

    return dynd_ndt_type_from_cpp(make_type[tuple_type]())

from libcpp.string cimport string

def struct(**kwds):
    # TODO: require an ordered dict here since struct types are ordered.
    # TODO: Use something other than dynd arrays to pass these arguments.
    #       See the comment in the tuple function for details.
    cdef vector[_type] _kwds
    cdef vector[string] _names

    if kwds:
        for kwd in kwds.values():
            _kwds.push_back(as_cpp_type(kwd))

        for name in kwds.keys():
            _names.push_back(name)

        return dynd_ndt_type_from_cpp(make_type[struct_type](_names, _kwds))

    return dynd_ndt_type_from_cpp(make_type[struct_type]())

#class fixed_dim(__builtins__.type):
#    def __call__(self, size_or_element_tp, element_tp = None):
#        if isinstance(size_or_element_tp, type):
#            return make_fixed_dim_kind(size_or_element_tp)
#
#        return make_fixed_dim(size_or_element_tp, (<type> element_tp).v)

def callable(ret_or_func, *args, **kwds):
    if isinstance(ret_or_func, type):
        ret = ret_or_func
    else:
        func = ret_or_func
        try:
            ret = func.__annotations__['return']
        except (AttributeError, KeyError):
            ret = type('Scalar')
        args = []
        for name in func.__code__.co_varnames:
            try:
                args.append(func.__annotations__[name])
            except (AttributeError, KeyError):
                args.append(type('Scalar'))
    return dynd_ndt_type_from_cpp(make_type[_callable_type](as_cpp_type(ret),
               as_cpp_type(tuple(*args)), as_cpp_type(struct(**kwds))))

"""
class struct_factory(__builtins__.type):
    def __call__(self, **kwds):
        make_struct(asarray(kwds.keys()).v, asarray(kwds.values()).v)
        return None
#        return wrap(make_tuple(asarray(args).v))

class struct(object):
    __metaclass__ = struct_factory
"""

scalar = type('Scalar')

_to_numba_type = {}
_from_numba_type = {}

try:
    import numba

    _to_numba_type[bool_id] = numba.boolean
    _to_numba_type[int8_id] = numba.int8
    _to_numba_type[int16_id] = numba.int16
    _to_numba_type[int32_id] = numba.int32
    _to_numba_type[int64_id] = numba.int64
    _to_numba_type[uint8_id] = numba.uint8
    _to_numba_type[uint16_id] = numba.uint16
    _to_numba_type[uint32_id] = numba.uint32
    _to_numba_type[uint64_id] = numba.uint64
    _to_numba_type[float32_id] = numba.float32
    _to_numba_type[float64_id] = numba.float64
    _to_numba_type[complex_float32_id] = numba.complex64
    _to_numba_type[complex_float64_id] = numba.complex128

    _from_numba_type = dict((_to_numba_type[key], key) for key in _to_numba_type)
except ImportError:
    pass

cdef as_numba_type(_type tp):
    return _to_numba_type[tp.get_id()]

cdef _type from_numba_type(tp):
    return _type(<type_id_t> _from_numba_type[tp])

cdef _type cpp_type_for(object obj) except *:
    cdef _type tp = _xtype_for_prefix(obj)
    if (not tp.is_null() and not isinstance(obj, _np.integer)):
        return tp
    if _builtin_type(obj) is builtin_tuple:
        obj = list(obj)
    if _builtin_type(obj) is list:
        return ndt_type_from_pylist(obj)
    tp = cpp_type_from_typeobject(_builtin_type(obj))
    return tp

def type_for(obj):
    return dynd_ndt_type_from_cpp(cpp_type_for(obj))

# Avoid circular import issues by importing these last.
from ..nd.array cimport (_numpy_dtype_from__type, _is_numpy_dtype, _xtype_for_prefix)

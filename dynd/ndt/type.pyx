# cython: c_string_type=str, c_string_encoding=ascii

from cpython.object cimport Py_EQ, Py_NE
from libc.stdint cimport intptr_t
from libcpp.string cimport string

from ..cpp.types.type_id cimport (type_id_t, uninitialized_type_id,
                                  bool_type_id, int8_type_id, int16_type_id,
                                  int32_type_id, int64_type_id, int128_type_id,
                                  uint8_type_id, uint16_type_id, uint32_type_id,
                                  uint64_type_id, uint128_type_id,
                                  float16_type_id, float32_type_id,
                                  float64_type_id, float128_type_id,
                                  complex_float32_type_id,
                                  complex_float64_type_id, void_type_id,
                                  callable_type_id)
from ..cpp.types.datashape_formatter cimport format_datashape as dynd_format_datashape
from ..cpp.types.byteswap_type cimport dynd_make_byteswap_type
from ..cpp.types.categorical_type cimport dynd_make_categorical_type
from ..cpp.types.type_alignment cimport make_unaligned as dynd_make_unaligned_type
from ..cpp.types.fixed_bytes_type cimport make_fixed_bytes as dynd_make_fixed_bytes_type
from ..cpp.types.fixed_dim_kind_type cimport dynd_make_fixed_dim_kind_type
from ..cpp.types.var_dim_type cimport dynd_make_var_dim_type
from ..cpp.types.tuple_type cimport make_tuple as _make_tuple
from ..cpp.types.struct_type cimport make_struct as _make_struct
from ..cpp.types.callable_type cimport make_callable

from ..config cimport translate_exception
from ..wrapper cimport set_wrapper_type, wrap

cdef extern from "numpy_interop.hpp" namespace "pydynd":
    object numpy_dtype_obj_from__type(_type&) except +translate_exception

cdef extern from 'gfunc_callable_functions.hpp' namespace 'pydynd':
    void add_type_names_to_dir_dict(_type&, object) except +translate_exception
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

__all__ = ['type_ids', 'type', 'bool', 'int8', 'int16', 'int32', 'int64', 'int128', \
    'uint8', 'uint16', 'uint32', 'uint64', 'uint128', 'float16', 'float32', \
    'float64', 'float128', 'complex_float32', 'complex_float64', 'void', \
    'tuple', 'struct', 'callable', 'scalar']

type_ids = {}
type_ids['UNINITIALIZED'] = uninitialized_type_id
type_ids['BOOL'] = bool_type_id
type_ids['INT8'] = int8_type_id
type_ids['INT16'] = int16_type_id
type_ids['INT32'] = int32_type_id
type_ids['INT64'] = int64_type_id
type_ids['INT128'] = int128_type_id
type_ids['UINT8'] = uint8_type_id
type_ids['UINT16'] = uint16_type_id
type_ids['UINT32'] = uint32_type_id
type_ids['UINT64'] = uint64_type_id
type_ids['UINT128'] = uint128_type_id
type_ids['FLOAT16'] = float16_type_id
type_ids['FLOAT32'] = float32_type_id
type_ids['FLOAT64'] = float64_type_id
type_ids['FLOAT128'] = float128_type_id
type_ids['COMPLEX64'] = complex_float32_type_id
type_ids['COMPLEX128'] = complex_float64_type_id
type_ids['VOID'] = void_type_id
type_ids['CALLABLE'] = callable_type_id

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
        system such as NumPy or ctypes.

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

    def __cinit__(self, rep=None):
        if rep is not None:
            self.v = make__type_from_pyobject(rep)

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

    property kind:
        """
        tp.kind
        The kind of this dynd type, as a string.
        Example kinds are 'bool', 'int', 'uint',
        'real', 'complex', 'string', 'uniform_array',
        'expression'.
        """
        def __get__(self):
            return _type_get_kind(self.v)

    property type_id:
        """
        tp.type_id
        The type id of this dynd type, as a string.
        Example type ids are 'bool', 'int8', 'uint32',
        'float64', 'complex_float32', 'string', 'byteswap'.
        """
        def __get__(self):
            return _type_get_type_id(self.v)

    def __getattr__(self, name):
        return get__type_dynamic_property(self.v, name)

    def __str__(self):
        return str(<char *>_type_str(self.v).c_str())

    def __repr__(self):
        return str(<char *>_type_repr(self.v).c_str())

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
        return numpy_dtype_obj_from__type(self.v)

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

set_wrapper_type[_type](type)

cdef class type_callable:
    cdef _type_callable_wrapper v

    def __call__(self, *args, **kwargs):
        return _type_callable_call(self.v, args, kwargs)

init_w__type_callable_typeobject(type_callable)

class UnsuppliedType(object):
    pass

Unsupplied = UnsuppliedType()

def make_byteswap(builtin_type, operand_type=None):
    """
    ndt.make_byteswap(builtin_type, operand_type=None)
    Constructs a byteswap type from a builtin one, with an
    optional expression type to chain in as the operand.
    Parameters
    ----------
    builtin_type : dynd type
        The builtin dynd type (like ndt.int16, ndt.float64) to
        which to apply the byte swap operation.
    operand_type: dynd type, optional
        An expression dynd type whose value type is a fixed bytes
        dynd type with the same data size and alignment as
        'builtin_type'.
    Examples
    --------
    >>> from dynd import nd, ndt
    >>> ndt.make_byteswap(ndt.int16)
    ndt.type("byteswap[int16]")
    """
    cdef type result = type()
    if operand_type is None:
        result.v = dynd_make_byteswap_type(type(builtin_type).v)
    else:
        result.v = dynd_make_byteswap_type(type(builtin_type).v, type(operand_type).v)
    return result

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

def make_convert(to_tp, from_tp):
    """
    ndt.make_convert(to_tp, from_tp)
    Constructs an expression type which converts from one
    dynd type to another.
    Parameters
    ----------
    to_tp : dynd type
        The dynd type being converted to. This is the 'value_type'
        of the resulting expression dynd type.
    from_tp : dynd type
        The dynd type being converted from. This is the 'operand_type'
        of the resulting expression dynd type.
    Examples
    --------
    >>> from dynd import nd, ndt
    >>> ndt.make_convert(ndt.int16, ndt.float32)
    ndt.type("convert[to=int16, from=float32]")
    """
    cdef type result = type()
    result.v = dynd_make_convert_type(type(to_tp).v, type(from_tp).v)
    return result

def make_unaligned(aligned_tp):
    """
    ndt.make_unaligned(aligned_tp)
    Constructs a type with alignment of 1 from the given type.
    If the type already has alignment 1, just returns it.
    Parameters
    ----------
    aligned_tp : dynd type
        The dynd type which should be viewed on data that is
        not properly aligned.
    Examples
    --------
    >>> from dynd import nd, ndt
    >>> ndt.make_unaligned(ndt.int32)
    ndt.type("unaligned[int32]")
    >>> ndt.make_unaligned(ndt.uint8)
    ndt.uint8
    """
    cdef type result = type()
    result.v = dynd_make_unaligned_type(type(aligned_tp).v)
    return result

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

def make_string(encoding=None):
    """
    ndt.make_string(encoding='utf_8')
    Constructs a variable-sized string dynd type
    with the specified encoding.
    Parameters
    ----------
    encoding : string, optional
        The encoding used for storing unicode code points. Supported
        values are 'ascii', 'utf_8', 'utf_16', 'utf_32', 'ucs_2'.
        Default: 'utf_8'.
    Examples
    --------
    >>> from dynd import nd, ndt
    >>> ndt.make_string()
    ndt.string
    >>> ndt.make_string('utf_16')
    ndt.type("string['utf16']")
    """
    cdef type result = type()
    result.v = dynd_make_string_type(encoding)
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
    result.v = dynd_make_var_dim_type(type(element_tp).v)
    return result


def make_view(value_type, operand_type):
    """
    ndt.make_view(value_type, operand_type)
    Constructs an expression type which views the bytes of
    one type as another.
    Parameters
    ----------
    value_type : dynd type
        The dynd type to interpret the bytes as. This is the 'value_type'
        of the resulting expression dynd type.
    operand_type : dynd type
        The dynd type the memory originally was. This is the 'operand_type'
        of the resulting expression dynd type.
    Examples
    --------
    >>> from dynd import nd, ndt
    >>> ndt.make_view(ndt.int32, ndt.uint32)
    ndt.type("view[as=int32, original=uint32]")
    """
    cdef type result = type()
    result.v = dynd_make_view_type(type(value_type).v, type(operand_type).v)
    return result

bool = type(bool_type_id)
int8 = type(int8_type_id)
int16 = type(int16_type_id)
int32 = type(int32_type_id)
int64 = type(int64_type_id)
int128 = type(int128_type_id)
uint8 = type(uint8_type_id)
uint16 = type(uint16_type_id)
uint32 = type(uint32_type_id)
uint64 = type(uint64_type_id)
uint128 = type(uint128_type_id)
float16 = type(float16_type_id)
float32 = type(float32_type_id)
float64 = type(float64_type_id)
float128 = type(float128_type_id)
complex_float32 = type(complex_float32_type_id)
complex_float64 = type(complex_float64_type_id)
void = type(void_type_id)

class tuple_factory(__builtins__.type):
    def __call__(self, *args):
        from ..nd.array import asarray

        if args:
            return wrap(_make_tuple((<array> asarray(args)).v))

        return wrap(_make_tuple())

class tuple(object):
    __metaclass__ = tuple_factory

class struct_factory(__builtins__.type):
    def __call__(self, **kwds):
        from ..nd.array import asarray

        if kwds:
            return wrap(_make_struct((<array> asarray(list(kwds.keys()))).v, (<array> asarray(list(kwds.values()))).v))

        return wrap(_make_struct())

class struct(object):
    __metaclass__ = struct_factory

#class fixed_dim(__builtins__.type):
#    def __call__(self, size_or_element_tp, element_tp = None):
#        if isinstance(size_or_element_tp, type):
#            return make_fixed_dim_kind(size_or_element_tp)
#
#        return make_fixed_dim(size_or_element_tp, (<type> element_tp).v)

class callable_factory(__builtins__.type):
    def __call__(self, ret_or_func, *args, **kwds):
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
#            args = [func.__annotations__[name] for name in func.__code__.co_varnames]

        return wrap(make_callable((<type> ret).v, (<type> tuple(*args)).v, (<type> struct(**kwds)).v))

    @property
    def id(self):
        return callable_type_id

class callable(object):
    __metaclass__ = callable_factory

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

    _to_numba_type[bool_type_id] = numba.boolean
    _to_numba_type[int8_type_id] = numba.int8
    _to_numba_type[int16_type_id] = numba.int16
    _to_numba_type[int32_type_id] = numba.int32
    _to_numba_type[int64_type_id] = numba.int64
    _to_numba_type[uint8_type_id] = numba.uint8
    _to_numba_type[uint16_type_id] = numba.uint16
    _to_numba_type[uint32_type_id] = numba.uint32
    _to_numba_type[uint64_type_id] = numba.uint64
    _to_numba_type[float32_type_id] = numba.float32
    _to_numba_type[float64_type_id] = numba.float64
    _to_numba_type[complex_float32_type_id] = numba.complex64
    _to_numba_type[complex_float64_type_id] = numba.complex128

    _from_numba_type = dict((_to_numba_type[key], key) for key in _to_numba_type)
except ImportError:
    pass

cdef as_numba_type(_type tp):
    return _to_numba_type[tp.get_type_id()]

cdef _type from_numba_type(tp):
    return _type(<type_id_t> _from_numba_type[tp])

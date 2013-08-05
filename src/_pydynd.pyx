#
# Copyright (C) 2011-13 Mark Wiebe, DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

# cython: c_string_type=str, c_string_encoding=ascii

cdef extern from "exception_translation.hpp" namespace "pydynd":
    void translate_exception()
    void set_broadcast_exception(object)

# Exceptions to convert from C++
class BroadcastError(Exception):
    pass

# Register all the exception objects with the exception translator
set_broadcast_exception(BroadcastError)

# Initialize Numpy
cdef extern from "do_import_array.hpp":
    pass
cdef extern from "numpy_interop.hpp" namespace "pydynd":
    object array_as_numpy_struct_capsule(ndarray&) except +translate_exception
    void import_numpy()
import_numpy()

# Initialize ctypes C level interop data
cdef extern from "ctypes_interop.hpp" namespace "pydynd":
    void init_ctypes_interop() except +translate_exception
init_ctypes_interop()

# Initialize C++ access to the Cython type objects
init_w_array_typeobject(w_array)
init_w_type_typeobject(w_type)
init_w_array_callable_typeobject(w_array_callable)
init_w_ndt_type_callable_typeobject(w_type_callable)

include "dynd.pxd"
#include "codegen_cache.pxd"
include "ndt_type.pxd"
include "array.pxd"
include "elwise_gfunc.pxd"
include "elwise_reduce_gfunc.pxd"
include "vm_elwise_program.pxd"
include "gfunc_callable.pxd"

# Issue a performance warning if any of the diagnostics macros are enabled
cdef extern from "<dynd/diagnostics.hpp>" namespace "dynd":
    bint any_diagnostics_enabled()
    string which_diagnostics_enabled()
if any_diagnostics_enabled():
    import warnings
    class PerformanceWarning(Warning):
        pass
    warnings.warn("Performance is reduced because of enabled diagnostics:\n" +
                str(<char *>which_diagnostics_enabled().c_str()), PerformanceWarning)

from cython.operator import dereference
# Save the built-in type operator, so we can have parameters called 'type'
builtin_type = type

# Create the codegen cache used by default when making gfuncs
#cdef w_codegen_cache default_cgcache_c = w_codegen_cache()
# Expose it outside the module too
#default_cgcache = default_cgcache_c

# Expose the git hashes and version numbers of this build
# NOTE: Cython generates code which is not const-correct, so
#       have to cast it away.
_dynd_version_string = str(<char *>dynd_version_string)
_dynd_git_sha1 = str(<char *>dynd_git_sha1)
_dynd_python_version_string = str(<char *>dynd_python_version_string)
_dynd_python_git_sha1 = str(<char *>dynd_python_git_sha1)

def _get_lowlevel_api():
    return <size_t>dynd_get_lowlevel_api()

def _get_py_lowlevel_api():
    return <size_t>dynd_get_py_lowlevel_api()

cdef class w_type:
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
    >>> ndt.type('5, var, float32')
    ndt.type('fixed_dim<5, var_dim<float32>>')
    >>> ndt.type('{x: float32; y: float32; z: float32}')
    ndt.type('cstruct<float32 x, float32 y, float32 z>')
    """
    # To access the embedded ndt::type, use "GET(self.v)",
    # which returns a reference to the ndt::type, and
    # SET(self.v, <ndt::type value>), which sets the embedded
    # ndt::type's value.
    cdef ndt_type_placement_wrapper v

    def __cinit__(self, rep=None):
        placement_new(self.v)
        if rep is not None:
            SET(self.v, make_ndt_type_from_pyobject(rep))
    def __dealloc__(self):
        placement_delete(self.v)

    def __dir__(self):
        # Customize dir() so that additional properties of various types
        # will show up in IPython tab-complete, for example.
        result = dict(w_type.__dict__)
        result.update(object.__dict__)
        add_ndt_type_names_to_dir_dict(GET(self.v), result)
        return result.keys()

    def __call__(self, *args, **kwargs):
        return call_ndt_type_constructor_function(GET(self.v), args, kwargs)

    def __getattr__(self, name):
        return get_ndt_type_dynamic_property(GET(self.v), name)

    property dshape:
        """
        a.dshape

        The blaze datashape of the dynd type, as a string.
        """
        def __get__(self):
            return str(<char *>dynd_format_datashape(GET(self.v)).c_str())

    property data_size:
        """
        a.data_size

        The size, in bytes, of the data for an instance
        of this dynd type.
        """
        def __get__(self):
            return GET(self.v).get_data_size()

    property data_alignment:
        """
        a.data_alignment

        The alignment, in bytes, of the data for an
        instance of this dynd type.

        Data for this dynd type must always be aligned
        according to this alignment, unaligned data
        requires an adapter transformation applied.
        """
        def __get__(self):
            return GET(self.v).get_data_alignment()

    property metadata_size:
        """
        a.metadata_size

        The size, in bytes, of the metadata for
        this dynd type.
        """
        def __get__(self):
            return GET(self.v).get_metadata_size()

    property kind:
        """
        a.kind

        The kind of this dynd type, as a string.

        Example kinds are 'bool', 'int', 'uint',
        'real', 'complex', 'string', 'uniform_array',
        'expression'.
        """
        def __get__(self):
            return ndt_type_get_kind(GET(self.v))

    property type_id:
        """
        a.type_id

        The type id of this dynd type, as a string.

        Example type ids are 'bool', 'int8', 'uint32',
        'float64', 'complex_float32', 'string', 'byteswap'.
        """
        def __get__(self):
            return ndt_type_get_type_id(GET(self.v))

    property ndim:
        """
        a.ndim

        The number of array dimensions in this dynd type.

        This property is like NumPy
        ndarray's 'ndim'. Indexing with [] can in many cases
        go deeper than just the array dimensions, for
        example structs can be indexed this way.
        """
        def __get__(self):
            return GET(self.v).get_ndim()

    property dtype:
        """
        a.dtype

        The dynd type of the element after the 'ndim'
        array dimensions are indexed away.

        This property is roughly equivalent to NumPy
        ndarray's 'dtype'.
        """
        def __get__(self):
            cdef w_type result = w_type()
            SET(result.v, GET(self.v).get_dtype())
            return result;

    property value_type:
        """
        a.value_type

        If this is an expression dynd type, returns the
        dynd type that values after evaluation have. Otherwise,
        returns this dynd type unchanged.
        """
        def __get__(self):
            cdef w_type result = w_type()
            SET(result.v, GET(self.v).value_type())
            return result

    property operand_type:
        """
        a.operand_type

        If this is an expression dynd type, returns the
        dynd type that inputs to its expression evaluation
        have. Otherwise, returns this dynd type unchanged.
        """
        def __get__(self):
            cdef w_type result = w_type()
            SET(result.v, GET(self.v).operand_type())
            return result

    property canonical_type:
        """
        a.canonical_type

        Returns a version of this type that is canonical,
        where any intermediate pointers are removed and expressions
        are stripped away.
        """
        def __get__(self):
            cdef w_type result = w_type()
            SET(result.v, GET(self.v).get_canonical_type())
            return result

    property property_names:
        """
        a.property_names

        Returns the names of properties exposed by dynd arrays
        of this type.
        """
        def __get__(self):
            return ndt_type_array_property_names(GET(self.v))

    def __getitem__(self, x):
        cdef w_type result = w_type()
        SET(result.v, ndt_type_getitem(GET(self.v), x))
        return result

    def __str__(self):
        return str(<char *>ndt_type_str(GET(self.v)).c_str())

    def __repr__(self):
        return str(<char *>ndt_type_repr(GET(self.v)).c_str())

    def __richcmp__(lhs, rhs, int op):
        if op == Py_EQ:
            if type(lhs) == w_type and type(rhs) == w_type:
                return GET((<w_type>lhs).v) == GET((<w_type>rhs).v)
            else:
                return False
        elif op == Py_NE:
            if type(lhs) == w_type and type(rhs) == w_type:
                return GET((<w_type>lhs).v) != GET((<w_type>rhs).v)
            else:
                return False
        return NotImplemented

def replace_dtype(w_type dt, replacement_dt, size_t replace_ndim=0):
    """
    replace_dtype(dt, replacement_dt, replace_ndim=0)

    Replaces the dtype with the replacement.
    If `replace_ndim` is positive, that number of uniform
    dimensions are replaced as well.

    Parameters
    ----------
    dt : dynd type
        The dtype whose dtype is to be replaced.
    replacement_dt : dynd type
        The replacement dynd type.
    replace_ndim : integer, optional
        If positive, this is the number of array dimensions
        which are included in the dtype for replacement.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> d = ndt.type('3, var, int32')
    >>> ndt.replace_dtype(d, 'M, float64')
    ndt.type('fixed_dim<3, var_dim<strided_dim<float64>>>')
    >>> ndt.replace_dtype(d, '{x: int32; y:int32}', 1)
    ndt.type('fixed_dim<3, cstruct<int32 x, int32 y>>')
    """
    cdef w_type result = w_type()
    SET(result.v, GET(dt.v).with_replaced_dtype(GET(w_type(replacement_dt).v), replace_ndim))
    return result

def extract_dtype(dt, size_t include_ndim=0):
    """
    extract_dtype(dt, include_ndim=0)

    Extracts the uniform type from the provided
    dynd type. If `keep_ndim` is positive, that
    many array dimensions are kept in the result.

    Parameters
    ----------
    dt : dynd type
        The dtype whose dtype is to be extracted.
    keep_ndim : integer, optional
        If positive, this is the number of array dimensions
        which are extracted in the dtype for replacement.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> d = ndt.type('3, var, int32')
    >>> ndt.extract_dtype(d)
    ndt.int32
    >>> ndt.extract_dtype(d, 1)
    ndt.type('var_dim<int32>')
    """
    cdef w_type result = w_type()
    SET(result.v, GET(w_type(dt).v).get_dtype(include_ndim))
    return result

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
    ndt.type('byteswap<int16>')
    """
    cdef w_type result = w_type()
    if operand_type is None:
        SET(result.v, dynd_make_byteswap_type(GET(w_type(builtin_type).v)))
    else:
        SET(result.v, dynd_make_byteswap_type(GET(w_type(builtin_type).v), GET(w_type(operand_type).v)))
    return result

def make_fixedbytes(int data_size, int data_alignment=1):
    """
    ndt.make_fixedbytes(data_size, data_alignment=1)

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

    >>> ndt.make_fixedbytes(4)
    ndt.type('fixedbytes<4,1>')
    >>> ndt.make_fixedbytes(6, 2)
    ndt.type('fixedbytes<6,2>')
    """
    cdef w_type result = w_type()
    SET(result.v, dynd_make_fixedbytes_type(data_size, data_alignment))
    return result

def make_convert(to_tp, from_tp, errmode=None):
    """
    ndt.make_convert(to_tp, from_tp, errmode='fractional')

    Constructs an expression type which converts from one
    dynd type to another, using a specified mode for handling
    conversion errors.

    Parameters
    ----------
    to_tp : dynd type
        The dynd type being converted to. This is the 'value_type'
        of the resulting expression dynd type.
    from_tp : dynd type
        The dynd type being converted from. This is the 'operand_type'
        of the resulting expression dynd type.
    errmode : 'inexact', 'fractional', 'overflow', 'none'
        How conversion errors are treated. For 'inexact', the value
        must be preserved precisely. For 'fractional', conversion errors
        due to precision loss are accepted, but not for loss of the
        fraction part. For 'overflow', only overflow errors are raised.
        For 'none', no conversion errors are raised

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> ndt.make_convert(ndt.int16, ndt.float32)
    ndt.type('convert<to=int16, from=float32>')
    >>> ndt.make_convert(ndt.uint8, ndt.uint16, 'none')
    ndt.type('convert<to=uint8, from=uint16, errmode=none>')
    """
    cdef w_type result = w_type()
    SET(result.v, dynd_make_convert_type(GET(w_type(to_tp).v), GET(w_type(from_tp).v), errmode))
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
    ndt.type('view<as=int32, original=uint32>')
    """
    cdef w_type result = w_type()
    SET(result.v, dynd_make_view_type(GET(w_type(value_type).v), GET(w_type(operand_type).v)))
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
    ndt.type('unaligned<int32>')
    >>> ndt.make_unaligned(ndt.uint8)
    ndt.uint8
    """
    cdef w_type result = w_type()
    SET(result.v, dynd_make_unaligned_type(GET(w_type(aligned_tp).v)))
    return result

def make_fixedstring(int size, encoding=None):
    """
    ndt.make_fixedstring(size, encoding='utf_8')

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

    >>> ndt.make_fixedstring(10)
    ndt.type('string<10>')
    >>> ndt.make_fixedstring(10, 'utf_32')
    ndt.type('string<10,utf_32>')
    """
    cdef w_type result = w_type()
    SET(result.v, dynd_make_fixedstring_type(size, encoding))
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
    ndt.type('string<utf_16>')
    """
    cdef w_type result = w_type()
    SET(result.v, dynd_make_string_type(encoding))
    return result

def make_bytes(size_t alignment=1):
    """
    ndt.make_bytes(alignment=1)

    Constructs a variable-sized bytes dynd type
    with the specified alignment.

    Parameters
    ----------
    alignment : int, optional
        The byte alignment of the raw binary data.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> ndt.make_bytes()
    ndt.bytes
    >>> ndt.make_string(4)
    ndt.type('bytes<align=4>')
    """
    cdef w_type result = w_type()
    SET(result.v, dynd_make_bytes_type(alignment))
    return result

def make_pointer(target_tp):
    """
    ndt.make_pointer(target_tp)

    Constructs a dynd type which is a pointer to the target type.

    Parameters
    ----------
    target_tp : dynd type
        The type that the pointer points to. This is similar to
        the '*' in C/C++ type declarations.
    """
    cdef w_type result = w_type()
    SET(result.v, dynd_make_pointer_type(GET(w_type(target_tp).v)))
    return result

def make_strided_dim(element_tp, ndim=None):
    """
    ndt.make_strided_dim(element_tp, ndim=1)

    Constructs an array dynd type with one or more strided
    dimensions. A single strided_dim dynd type corresponds
    to one dimension, so when ndim > 1, multiple strided_dim
    dimensions are created.

    Parameters
    ----------
    element_tp : dynd type
        The type of one element in the strided array.
    ndim : int
        The number of strided_dim dimensions to create.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> ndt.make_strided_dim(ndt.int32)
    ndt.type('strided_dim<int32>')
    >>> ndt.make_strided_dim(ndt.int32, 3)
    ndt.type('strided_dim<strided_dim<strided_dim<int32>>>')
    """
    cdef w_type result = w_type()
    if (ndim is None):
        SET(result.v, dynd_make_strided_dim_type(GET(w_type(element_tp).v)))
    else:
        SET(result.v, dynd_make_strided_dim_type(GET(w_type(element_tp).v), int(ndim)))
    return result

def make_fixed_dim(shape, element_tp, axis_perm=None):
    """
    ndt.make_fixed_dim(shape, element_tp, axis_perm=None)

    Constructs a fixed_dim type of the given shape and axis permutation
    (default C order).

    Parameters
    ----------
    shape : tuple of int
        The multi-dimensional shape of the resulting fixed array type.
    element_tp : dynd type
        The type of each element in the resulting array type.
    axis_perm : tuple of int
        If not provided, C-order is used. Must be a permutation of
        the integers 0 through len(shape)-1, ordered so each
        value increases with the size of the strides. [N-1, ..., 0]
        gives C-order, and [0, ..., N-1] gives F-order.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> ndt.make_fixed_dim(5, ndt.int32)
    ndt.type('fixed_dim<5, int32>')
    >>> ndt.make_fixed_dim((3,5), ndt.int32)
    ndt.type('fixed_dim<3, fixed_dim<5, int32>>')
    >>> ndt.make_fixed_dim((3,5), ndt.int32, axis_perm=(0,1))
    ndt.type('fixed_dim<3, stride=4, fixed_dim<5, stride=12, int32>>')
    """
    cdef w_type result = w_type()
    SET(result.v, dynd_make_fixed_dim_type(shape, GET(w_type(element_tp).v), axis_perm))
    return result

def make_cstruct(field_types, field_names):
    """
    ndt.make_cstruct(field_types, field_names)

    Constructs a fixed_struct dynd type, which has fields with
    a fixed layout.

    The fields are laid out in memory in the order they
    are specified, each field aligned as required
    by its type, and the total data size padded so that adjacent
    instances of the type are properly aligned.

    Parameters
    ----------
    field_types : list of dynd types
        A list of types, one for each field.
    field_names : list of strings
        A list of names, one for each field, corresponding to 'field_types'.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> ndt.make_cstruct([ndt.int32, ndt.float64], ['x', 'y'])
    ndt.type('cstruct<int32 x, float64 y>')
    """
    cdef w_type result = w_type()
    SET(result.v, dynd_make_cstruct_type(field_types, field_names))
    return result

def make_struct(field_types, field_names):
    """
    ndt.make_struct(field_types, field_names)

    Constructs a struct dynd type, which has fields with a flexible
    per-array layout.

    If a subset of fields from a fixed_struct are taken,
    the result is a struct, with the layout specified
    in the dynd array's metadata.

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
    ndt.type('struct<int32 x, float64 y>')
    """
    cdef w_type result = w_type()
    SET(result.v, dynd_make_struct_type(field_types, field_names))
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
    ndt.type('var_dim<float32>')
    """
    cdef w_type result = w_type()
    SET(result.v, dynd_make_var_dim_type(GET(w_type(element_tp).v)))
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
    ndt.type('categorical<string<ascii>, ["sunny", "rainy", "cloudy", "stormy"]>')
    """
    cdef w_type result = w_type()
    SET(result.v, dynd_make_categorical_type(GET(w_array(values).v)))
    return result

def factor_categorical(values):
    """
    ndt.factor_categorical(values)

    Constructs a categorical dynd type with the
    unique sorted subset of the values as its
    categories.

    Parameters
    ----------
    values : one-dimensional dynd array
        This is an array of the values that are sorted, with
        duplicates removed, to produce the categories of
        the resulting dynd type.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> ndt.factor_categorical(['M', 'M', 'F', 'F', 'M', 'F', 'M'])
    ndt.type('categorical<string<ascii>, ["F", "M"]>')
    """
    cdef w_type result = w_type()
    SET(result.v, dynd_factor_categorical_type(GET(w_array(values).v)))
    return result

##############################################################################

# NOTE: This is a possible alternative to the init_w_array_typeobject() call
#       above, but it generates a 1300 line header file and still requires calling
#       import__dnd from the C++ code, so directly using C++ primitives seems simpler.
#cdef public api class w_array [object WNDArrayObject, type WNDArrayObject_Type]:

cdef class w_array:
    """
    nd.array(obj=None, dtype=None, type=None, access=None)

    Create a dynd array out of the provided object.

    The dynd array is the dynamically typed multi-dimensional
    object provided by the dynd library. It is similar to
    NumPy's ndarray, but has its dimensional structure encoded
    in the dynd type, along with the element type.

    When given a NumPy array, the resulting dynd array is a view
    into the NumPy array data. When given lists of Python object,
    an attempt is made to deduce an appropriate dynd type for
    the array, and a conversion is made if possible, or an
    exception is raised.

    Parameters
    ----------
    obj : multi-dimensional object, optional
        Any object which dynd knows how to interpret as a dynd array.
    dtype: dynd type
        If provided, the type is used as the data type for the
        input, and the shape of the leading dimensions is deduced.
        This parameter cannot be used together with 'type'.
    type: dynd type
        If provided, the type is used as the full type for the input.
        If needed by the type, the shape is deduced from the input.
        This parameter cannot be used together with 'dtype'.
    access:  'readwrite'/'rw', 'readonly'/'r', or 'immutable', optional
        If provided, this specifies the access control for the
        created array. If the array is being allocated, as in
        construction from Python objects, this is the access control
        set.

        If the array is a view into another array data source,
        such as NumPy arrays or objects which support the buffer
        protocol, the access control must be compatible with that
        of the input object's.

        The default is immutable, or to inherit the access control
        of the object being viewed if it is an object supporting
        the buffer protocol.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> nd.array([1, 2, 3, 4, 5])
    nd.array([1, 2, 3, 4, 5], strided_dim<int32>)
    >>> nd.array([[1, 2], [3, 4, 5.0]])
    nd.array([[1, 2], [3, 4, 5]], strided_dim<var_dim<float64>>)
    >>> from datetime import date
    >>> nd.array([date(2000,2,14), date(2012,1,1)])
    nd.array([2000-02-14, 2012-01-01], strided_dim<date>)
    """
    # To access the embedded ndt::type, use "GET(self.v)",
    # which returns a reference to the dynd array, and
    # SET(self.v, <array value>), which sets the embeded
    # array's value.
    cdef array_placement_wrapper v

    def __cinit__(self, obj=None, dtype=None, type=None, access=None):
        placement_new(self.v)
        if obj is not None:
            # Get the array data
            if dtype is not None:
                if type is not None:
                    raise ValueError('Must provide only one of ' +
                                    'dtype or type, not both')
                array_init_from_pyobject(GET(self.v), obj, dtype, True, access)
            elif type is not None:
                array_init_from_pyobject(GET(self.v), obj, type, False, access)
            else:
                array_init_from_pyobject(GET(self.v), obj, access)

    def __dealloc__(self):
        placement_delete(self.v)

    def __dir__(self):
        # Customize dir() so that additional properties of various types
        # will show up in IPython tab-complete, for example.
        result = dict(w_array.__dict__)
        result.update(object.__dict__)
        add_array_names_to_dir_dict(GET(self.v), result)
        return result.keys()

    def __getattr__(self, name):
        return get_array_dynamic_property(GET(self.v), name)

    def __setattr__(self, name, value):
        set_array_dynamic_property(GET(self.v), name, value)

    def __contains__(self, x):
        return array_contains(GET(self.v), x)

    def eval(self):
        """
        a.eval()

        Returns a version of the dynd array with plain values,
        all expressions evaluated. This returns the original
        array back if it has no expression type.

        Examples
        --------
        >>> from dynd import nd, ndt

        >>> a = nd.array([1.5, 2, 3])
        >>> a
        nd.array([1.5, 2, 3], strided_dim<float64>)
        >>> b = a.ucast(ndt.int16, errmode='none')
        >>> b
        nd.array([1, 2, 3], strided_dim<convert<to=int16, from=float64, errmode=none>>)
        >>> b.eval()
        nd.array([1, 2, 3], strided_dim<int16>)
        """
        cdef w_array result = w_array()
        SET(result.v, array_eval(GET(self.v)))
        return result

    def eval_immutable(self):
        """
        a.eval_immutable()

        Evaluates into an immutable dynd array. If the array is
        already immutable and not an expression type, returns it
        as is.
        """
        cdef w_array result = w_array()
        SET(result.v, GET(self.v).eval_immutable())
        return result

    def eval_copy(self, access=None):
        """
        a.eval_copy(access='readwrite')

        Evaluates into a new dynd array, guaranteeing a copy is made.

        Parameters
        ----------
        access : 'readwrite' or 'immutable'
            Specifies the access control of the resulting copy.
        """
        cdef w_array result = w_array()
        SET(result.v, array_eval_copy(GET(self.v), access))
        return result

    def storage(self):
        """
        a.storage()

        Returns a version of the dynd array with its storage type,
        all expressions discarded. For data types that are plain
        old data, views them as a bytes type.

        Examples
        --------
        >>> from dynd import nd, ndt

        >>> a = nd.array([1, 2, 3], type=ndt.int16)
        >>> a
        nd.array([1, 2, 3], strided_dim<int16>)
        >>> a.storage()
        nd.array([0x0100, 0x0200, 0x0300], strided_dim<fixedbytes<2,2>>)
        """
        cdef w_array result = w_array()
        SET(result.v, GET(self.v).storage())
        return result

    def cast(self, type, errmode=None):
        """
        a.cast(type, errmode='fractional')

        Casts the dynd array's type to the requested type,
        producing a conversion type. If the data for the
        new type is identical, it is used directly to avoid
        the conversion.

        Parameters
        ----------
        type : dynd type
            The type is cast into this type.
        errmode : 'inexact', 'fractional', 'overflow', 'none'
            How conversion errors are treated. For 'inexact', the value
            must be preserved precisely. For 'fractional', conversion errors
            due to precision loss are accepted, but not for loss of the
            fraction part. For 'overflow', only overflow errors are raised.
            For 'none', no conversion errors are raised

        """
        cdef w_array result = w_array()
        SET(result.v, array_cast(GET(self.v), GET(w_type(type).v), errmode))
        return result

    def ucast(self, dtype, int replace_ndim=0, errmode=None):
        """
        a.ucast(dtype, replace_ndim=0, errmode='fractional')

        Casts the dynd array's dtype to the requested type,
        producing a conversion type. The dtype is the type
        after the nd.ndim_of(a) array dimensions.

        Parameters
        ----------
        dtype : dynd type
            The dtype of the array is cast into this type.
            If `replace_ndim` is not zero, then that many
            dimensions are included in what is cast as well.
        replace_ndim : integer, optional
            The number of array dimensions to replace in doing
            the cast.
        errmode : 'inexact', 'fractional', 'overflow', 'none'
            How conversion errors are treated. For 'inexact', the value
            must be preserved precisely. For 'fractional', conversion errors
            due to precision loss are accepted, but not for loss of the
            fraction part. For 'overflow', only overflow errors are raised.
            For 'none', no conversion errors are raised

        Examples
        --------
        >>> from dynd import nd, ndt

        >>> from datetime import date
        >>> a = nd.array([date(1929,3,13), date(1979,3,22)]).ucast('{month: int32; year: int32; day: float32}')
        >>> a
        nd.array([[3, 1929, 13], [3, 1979, 22]], strided_dim<convert<to=cstruct<int32 month, int32 year, float32 day>, from=date>>)
        >>> a.eval()
        nd.array([[3, 1929, 13], [3, 1979, 22]], strided_dim<cstruct<int32 month, int32 year, float32 day>>)
        """
        cdef w_array result = w_array()
        SET(result.v, array_ucast(GET(self.v), GET(w_type(dtype).v), replace_ndim, errmode))
        return result

    def view_scalars(self, dtype):
        """
        a.view_scalars(dtype)

        Views the data of the dynd array as the requested dtype,
        where it makes sense.

        If the array is a one-dimensional contiguous
        array of plain old data, the new dtype may have a different
        element size than original one.

        When the array has an expression type, the
        view is created by layering another view dtype
        onto the array's existing expression.

        Parameters
        ----------
        dtype : dynd type
            The scalars are viewed as this dtype.
        """
        cdef w_array result = w_array()
        SET(result.v, GET(self.v).view_scalars(GET(w_type(dtype).v)))
        return result

    def flag_as_immutable(self):
        """
        a.flag_as_immutable()

        When there's still only one reference to a
        dynd array, can be used to flag it as immutable.
        """
        GET(self.v).flag_as_immutable()

    property access_flags:
        """
        a.access_flags

        The access flags of the dynd array, as a string.
        Returns 'immutable', 'readonly', or 'readwrite'
        """
        def __get__(self):
            return str(<char *>array_access_flags_string(GET(self.v)))

    property is_scalar:
        """
        a.is_scalar

        True if the dynd array is a scalar.
        """
        def __get__(self):
            return GET(self.v).is_scalar()

    property shape:
        def __get__(self):
            return array_get_shape(GET(self.v))

    property strides:
        def __get__(self):
            return array_get_strides(GET(self.v))

    def __str__(self):
        return array_str(GET(self.v))

    def __unicode__(self):
        return array_unicode(GET(self.v))

    def __index__(self):
        return array_index(GET(self.v))

    def __nonzero__(self):
        return array_nonzero(GET(self.v))

    def __repr__(self):
        return str(<char *>array_repr(GET(self.v)).c_str())

    def __len__(self):
        if GET(self.v).is_scalar():
            raise TypeError('zero-dimensional nd::array has no len()')
        return GET(self.v).get_dim_size()

    def __getitem__(self, x):
        cdef w_array result = w_array()
        SET(result.v, array_getitem(GET(self.v), x))
        return result

    def __setitem__(self, x, y):
        array_setitem(GET(self.v), x, y)

    def __getbuffer__(w_array self, Py_buffer* buffer, int flags):
        # Docstring triggered Cython bug (fixed in master), so it's commented out
        #"""PEP 3118 buffer protocol"""
        array_getbuffer_pep3118(self, buffer, flags)

    def __releasebuffer__(w_array self, Py_buffer* buffer):
        # Docstring triggered Cython bug (fixed in master), so it's commented out
        #"""PEP 3118 buffer protocol"""
        array_releasebuffer_pep3118(self, buffer)

    def __add__(lhs, rhs):
        cdef w_array result = w_array()
        SET(result.v, array_add(GET(w_array(lhs).v), GET(w_array(rhs).v)))
        return result

    def __sub__(lhs, rhs):
        cdef w_array result = w_array()
        SET(result.v, array_subtract(GET(w_array(lhs).v), GET(w_array(rhs).v)))
        return result

    def __mul__(lhs, rhs):
        cdef w_array result = w_array()
        SET(result.v, array_multiply(GET(w_array(lhs).v), GET(w_array(rhs).v)))
        return result

    def __div__(lhs, rhs):
        cdef w_array result = w_array()
        SET(result.v, array_divide(GET(w_array(lhs).v), GET(w_array(rhs).v)))
        return result

def view(obj, access=None):
    """
    nd.view(obj, access=None)

    Constructs a dynd array which is a view of the data from
    `obj`. The `access` parameter can be used to require writable
    access for an output parameter, or to produce a read-only
    view of writable data.

    Parameters
    ----------
    obj : object
        A Python object which backs some array data, such as
        a dynd array, a numpy array, or an object supporting
        the Python buffer protocol.
    access : 'readwrite'/'rw' or 'readonly'/'r', optional
        The access flags for the constructed array. Use 'readwrite'
        to require that the view be writable, and 'readonly' to
        provide a view of data to someone else without allowing
        writing.
    """
    cdef w_array result = w_array()
    SET(result.v, array_view(obj, access))
    return result

def asarray(obj, access=None):
    """
    nd.asarray(obj, access=None)

    Constructs a dynd array from the object, taking a view
    if possible, otherwise making a copy.

    Parameters
    ----------
    obj : object
        The object which is to be converted into a dynd array,
        as a view if possible, otherwise a copy.
    access : 'readwrite'/'rw', 'readonly'/'r', 'immutable', optional
        If provided, the access flags the resulting array should
        satisfy. When a view can be taken, but these access flags
        cannot, a copy is made.
    """
    cdef w_array result = w_array()
    SET(result.v, array_asarray(obj, access))
    return result

def type_of(w_array a):
    """
    nd.type_of(a)

    The dynd type of the array. This is the full
    data type, including its multi-dimensional structure.

    Parameters
    ----------
    a : dynd array
        The array whose type is requested.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> nd.type_of(nd.array([1,2,3,4]))
    ndt.type('strided_dim<int32>')
    >>> nd.type_of(nd.array([[1,2],[3.0]]))
    ndt.type('strided_dim<var_dim<float64>>')
    """
    cdef w_type result = w_type()
    SET(result.v, GET(a.v).get_type())
    return result

def dshape_of(w_array a):
    """
    nd.dshape_of(a)

    The blaze datashape of the dynd array, as a string.

    Parameters
    ----------
    a : dynd array
        The array whose type is requested.
    """
    return str(<char *>dynd_format_datashape(GET(a.v)).c_str())

def dtype_of(w_array a, size_t include_ndim=0):
    """
    nd.dtype_of(a)

    The data type of the dynd array. This is
    the type after removing all the array
    dimensions from the dynd type of `a`.
    If `include_ndim` is provided, that many
    array dimensions are included in the
    data type returned.

    Parameters
    ----------
    a : dynd array
        The array whose type is requested.
    include_ndim : int, optional
        The number of array dimensions to include
        in the data type, default zero.
    """
    cdef w_type result = w_type()
    SET(result.v, GET(a.v).get_dtype(include_ndim))
    return result

def ndim_of(w_array a):
    """
    nd.ndim_of(a)

    The number of array dimensions in the dynd array `a`.
    This corresponds to the number of dimensions
    in a NumPy array.
    """
    return GET(a.v).get_ndim()

def as_py(w_array n):
    """
    nd.as_py(n)

    Evaluates the dynd array, converting it into native Python types.

    Uniform dimensions convert into Python lists, struct types convert
    into Python dicts, scalars convert into the most appropriate Python
    scalar for their type.

    Parameters
    ----------
    n : dynd array
        The dynd array to convert into native Python types.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> a = nd.array([1, 2, 3, 4.0])
    >>> a
    nd.array([1, 2, 3, 4], strided_dim<float64>)
    >>> nd.as_py(a)
    [1.0, 2.0, 3.0, 4.0]
    """
    return array_as_py(GET(n.v))

def as_numpy(w_array n, allow_copy=False):
    """
    nd.as_numpy(n, allow_copy=False)

    Evaluates the dynd array, and converts it into a NumPy object.

    Parameters
    ----------
    n : dynd array
        The array to convert into native Python types.
    allow_copy : bool, optional
        If true, allows a copy to be made when the array types
        can't be directly viewed as a NumPy array, but with a
        data-preserving copy they can be.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> import numpy as np
    >>> a = nd.array([[1, 2, 3], [4, 5, 6]])
    >>> a
    nd.array([[1, 2, 3], [4, 5, 6]], strided_dim<strided_dim<int32>>)
    >>> nd.as_numpy(a)
    array([[1, 2, 3],
           [4, 5, 6]])
    """
    # TODO: Could also convert dynd types into numpy dtypes
    return array_as_numpy(n, bool(allow_copy))

def empty(shape, type=None):
    """
    nd.empty(type)
    nd.empty(shape, type)

    Creates an uninitialized array of the specified
    shape if supplied, with the provided dynd type.

    Parameters
    ----------
    shape : list of int, optional
        If provided, specifies the shape for the type dimensions
        that don't encode a dimension size themselves, such as
        strided_dim dimensions.
    type : dynd type
        The data type of the uninitialized array to create. This
        is the full data type, including the multi-dimensional
        structure.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> nd.empty('2, 2, int8')
    nd.array([[0, -24], [0, 4]], fixed_dim<2, fixed_dim<2, int8>>)
    >>> nd.empty((2, 2), 'M, N, int16')
    nd.array([[179, 0], [0, 16816]], strided_dim<strided_dim<int16>>)
    """
    cdef w_array result = w_array()
    if type is not None:
        SET(result.v, array_empty(shape, GET(w_type(type).v)))
    else:
        # Interpret the first argument (shape) as a type in the one argument case
        SET(result.v, array_empty(GET(w_type(shape).v)))
    return result

def empty_like(w_array prototype, dtype=None):
    """
    nd.empty_like(prototype, dtype=None)

    Creates an uninitialized array whose array dimensions match
    the structure of the prototype's array dimensions.

    Parameters
    ----------
    prototype : dynd array
        The array whose structure is to be matched.
    dtype : dynd type, optional
        If provided, replaces the prototype's dtype in
        the result.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> a = nd.array([[1, 2], [3, 4]])
    >>> a
    nd.array([[1, 2], [3, 4]], strided_dim<strided_dim<int32>>)
    >>> nd.empty_like(a)
    nd.array([[808529973, 741351468], [0, 0]], strided_dim<strided_dim<int32>>)
    >>> nd.empty_like(a, dtype=ndt.float32)
    nd.array([[1.47949e-041, 0], [0, 0]], strided_dim<strided_dim<float32>>)
    """
    cdef w_array result = w_array()
    if dtype is None:
        SET(result.v, array_empty_like(GET(prototype.v)))
    else:
        SET(result.v, array_empty_like(GET(prototype.v), GET(w_type(dtype).v)))
    return result

def groupby(data, by, groups = None):
    """
    nd.groupby(data, by, groups=None)

    Produces an array containing the elements of `data`, grouped
    according to `by` which has corresponding shape.

    Parameters
    ----------
    data : dynd array
        A one-dimensional array of the data to be copied into
        the resulting grouped array.
    by : dynd array
        A one-dimensional array, of the same size as 'data',
        with the category values for the grouping.
    groups : categorical dynd type, optional
        If provided, the categories of this type are used
        as the groups for the grouping operation.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> a = nd.groupby([1, 2, 3, 4, 5, 6], ['M', 'F', 'M', 'M', 'F', 'F'])
    >>> a.groups
    nd.array(["F", "M"], strided_dim<string<ascii>>)
    >>> a.eval()
    nd.array([[2, 5, 6], [1, 3, 4]], fixed_dim<2, var_dim<int32>>)

    >>> a = nd.groupby([1, 2, 3, 4, 5, 6], ['M', 'F', 'M', 'M', 'F', 'F'], ['M', 'N', 'F'])
    >>> a.groups
    nd.array(["M", "N", "F"], strided_dim<string<ascii>>)
    >>> a.eval()
    nd.array([[1, 3, 4], [], [2, 5, 6]], fixed_dim<3, var_dim<int32>>)
    """
    cdef w_array result = w_array()
    if groups is None:
        SET(result.v, dynd_groupby(GET(w_array(data).v), GET(w_array(by).v)))
    else:
        if type(groups) in [list, w_array]:
            # If groups is a list or dynd array, assume it's a list
            # of groups for a categorical type
            SET(result.v, dynd_groupby(GET(w_array(data).v), GET(w_array(by).v),
                            dynd_make_categorical_type(GET(w_array(groups).v))))
        else:
            SET(result.v, dynd_groupby(GET(w_array(data).v), GET(w_array(by).v), GET(w_type(groups).v)))
    return result

def range(start=None, stop=None, step=None, dtype=None):
    """
    nd.range(stop, dtype=None)
    nd.range(start, stop, step=None, dtype=None)

    Constructs a dynd array representing a stepped range of values.

    This function assumes that (stop-start)/step is approximately
    an integer, so as to be able to produce reasonable values with
    fractional steps which can't be exactly represented, such as 0.1.

    Parameters
    ----------
    start : int, optional
        If provided, this is the first value in the resulting dynd array.
    stop : int
        This provides the stopping criteria for the range, and is
        not included in the resulting dynd array.
    step : int
        This is the increment.
    dtype : dynd type, optional
        If provided, it must be a scalar type, and the result
        is of this type.
    """
    cdef w_array result = w_array()
    # Move the first argument to 'stop' if stop isn't specified
    if stop is None:
        if start is not None:
            SET(result.v, array_range(None, start, step, dtype))
        else:
            raise ValueError("No value provided for 'stop'")
    else:
        SET(result.v, array_range(start, stop, step, dtype))
    return result

def linspace(start, stop, count=50, dtype=None):
    """
    nd.linspace(start, stop, count=50, dtype=None)

    Constructs a specified count of values interpolating a range.

    Parameters
    ----------
    start : floating point scalar
        The value of the first element of the resulting dynd array.
    stop : floating point scalar
        The value of the last element of the resulting dynd array.
    count : int, optional
        The number of elements in the resulting dynd array.
    dtype : dynd type, optional
        If provided, it must be a scalar type, and the result
        is of this type.
    """
    cdef w_array result = w_array()
    SET(result.v, array_linspace(start, stop, count, dtype))
    return result

def fields(w_array struct_array, *fields_list):
    """
    nd.fields(struct_array, *fields_list)

    Selects fields from an array of structs.

    Parameters
    ----------
    struct_array : dynd array with struct dtype
        A dynd array whose dtype has kind 'struct'. This
        could be a single struct instance, or an array of structs.
    *fields_list : string
        The remaining parameters must all be strings, and are the field
        names to select.
    """
    cdef w_array result = w_array()
    SET(result.v, nd_fields(GET(struct_array.v), fields_list))
    return result

def parse_json(type, json):
    """
    nd.parse_json(type, json)

    Parses an input JSON string as a particular dynd type.

    Parameters
    ----------
    type : dynd type
        The type to interpret the input JSON as. If the data
        does not match this type, an error is raised during parsing.
    json : string or bytes
        String that contains the JSON to parse.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> nd.parse_json('var, int8', '[1, 2, 3, 4, 5]')
    nd.array([1, 2, 3, 4, 5], var_dim<int8>)
    >>> nd.parse_json('4, int8', '[1, 2, 3, 4]')
    nd.array([1, 2, 3, 4], fixed_dim<4, int8>)
    >>> nd.parse_json('2, {x: int8; y: int8}', '[{"x":0, "y":1}, {"y":2, "x":3}]')
    nd.array([[0, 1], [3, 2]], fixed_dim<2, cstruct<int8 x, int8 y>>)
    """
    cdef w_array result = w_array()
    if builtin_type(type) is w_array:
        dynd_parse_json_array(GET((<w_array>type).v), GET(w_array(json).v))
    else:
        SET(result.v, dynd_parse_json_type(GET(w_type(type).v), GET(w_array(json).v)))
        return result

def format_json(w_array n):
    """
    nd.format_json(n)

    Formats a dynd array as JSON.

    Parameters
    ----------
    n : dynd array
        The object to format as JSON.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> a = nd.array([[1, 2, 3], [1, 2]])
    >>> a
    nd.array([[1, 2, 3], [1, 2]], strided_dim<var_dim<int32>>)
    >>> nd.format_json(a)
    nd.array("[[1,2,3],[1,2]]", string)
    """
    cdef w_array result = w_array()
    SET(result.v, dynd_format_json(GET(n.v)))
    return result

def elwise_map(n, callable, dst_type, src_type = None):
    """
    nd.elwise_map(n, callable, dst_type, src_type=None)

    Applies a deferred element-wise mapping function to
    a dynd array 'n'.

    Parameters
    ----------
    n : list of dynd array
        A list of objects to which the mapping is applied.
    callable : Python callable
        A Python function which is called as
        'callable(dst, src[0], ..., src[N-1])', with
        one-dimensional dynd arrays 'dst', 'src[0]',
        ..., 'src[N-1]'. The function should, in an element-wise
        fashion, use the values in the different 'src'
        arrays to calculate values that get placed into
        the 'dst' array. The function must not return a value.
    dst_type : dynd type
        The type of the computation's result.
    src_type : list of dynd type, optional
        A list of types of the source. If a source array has
        a different type than the one corresponding in this list,
        it will be converted.
    """
    return dynd_elwise_map(n, callable, dst_type, src_type)

class DebugReprObj(object):
    def __init__(self, repr_str):
        self.repr_str = repr_str

    def __str__(self):
        return self.repr_str

    def __repr__(self):
        return self.repr_str

def debug_repr(obj):
    """
    nd.debug_repr(a)

    Returns a raw representation of dynd array data.

    This can be useful for diagnosing bugs in the dynd array
    or type/metadata/data abstraction arrays are based on.

    Parameters
    ----------
    a : dynd array
        The object whose debug repr is desired
    """
    if isinstance(obj, w_array):
        return DebugReprObj(str(<char *>array_debug_print(GET((<w_array>obj).v)).c_str()))

cdef class w_elwise_gfunc:
    cdef elwise_gfunc_placement_wrapper v

    def __cinit__(self, bytes name):
        placement_new(self.v, name)
    def __dealloc__(self):
        placement_delete(self.v)

    property name:
        def __get__(self):
            return str(<char *>GET(self.v).get_name().c_str())

#    def add_kernel(self, kernel, w_codegen_cache cgcache = default_cgcache_c):
#        """Adds a kernel to the gfunc object. Currently, this means a ctypes object with prototype."""
#        elwise_gfunc_add_kernel(GET(self.v), GET(cgcache.v), kernel)

    #def debug_repr(self):
    #    """Prints a raw representation of the gfunc data."""
    #    return str(<char *>elwise_gfunc_debug_print(GET(self.v)).c_str())

    def __call__(self, *args, **kwargs):
        """Calls the gfunc."""
        return elwise_gfunc_call(GET(self.v), args, kwargs)

cdef class w_elwise_reduce_gfunc:
    cdef elwise_reduce_gfunc_placement_wrapper v

    def __cinit__(self, bytes name):
        placement_new(self.v, name)
    def __dealloc__(self):
        placement_delete(self.v)

    property name:
        def __get__(self):
            return str(<char *>GET(self.v).get_name().c_str())

#    def add_kernel(self, kernel, bint associative, bint commutative, identity = None, w_codegen_cache cgcache = default_cgcache_c):
#        """Adds a kernel to the gfunc object. Currently, this means a ctypes object with prototype."""
#        cdef w_array id
#        if identity is None:
#            elwise_reduce_gfunc_add_kernel(GET(self.v), GET(cgcache.v), kernel, associative, commutative, array())
#        else:
#            id = w_array(identity)
#            elwise_reduce_gfunc_add_kernel(GET(self.v), GET(cgcache.v), kernel, associative, commutative, GET(id.v))

    #def debug_repr(self):
    #    """Returns a raw representation of the gfunc data."""
    #    return str(<char *>elwise_reduce_gfunc_debug_print(GET(self.v)).c_str())

    def __call__(self, *args, **kwargs):
        """Calls the gfunc."""
        return elwise_reduce_gfunc_call(GET(self.v), args, kwargs)

#cdef class w_codegen_cache:
#    cdef codegen_cache_placement_wrapper v
#
#    def __cinit__(self):
#        placement_new(self.v)
#    def __dealloc__(self):
#        placement_delete(self.v)
#
#    def debug_repr(self):
#        """Prints a raw representation of the codegen_cache data."""
#        return str(<char *>codegen_cache_debug_print(GET(self.v)).c_str())

cdef class w_elwise_program:
    cdef vm_elwise_program_placement_wrapper v

    def __cinit__(self, obj=None):
        placement_new(self.v)
        if obj is not None:
            vm_elwise_program_from_py(obj, GET(self.v))
    def __dealloc__(self):
        placement_delete(self.v)

    def set(self, obj):
        """Sets the elementwise program from the provided dict"""
        vm_elwise_program_from_py(obj, GET(self.v))

    def as_dict(self):
        """Converts the elementwise VM program into a dict"""
        return vm_elwise_program_as_py(GET(self.v))

    #def debug_repr(self):
    #    """Returns a raw representation of the elwise_program data."""
    #    return str(<char *>vm_elwise_program_debug_print(GET(self.v)).c_str())

cdef class w_array_callable:
    cdef array_callable_placement_wrapper v

    def __cinit__(self):
        placement_new(self.v)
    def __dealloc__(self):
        placement_delete(self.v)

    def __call__(self, *args, **kwargs):
        return array_callable_call(GET(self.v), args, kwargs)

cdef class w_type_callable:
    cdef ndt_type_callable_placement_wrapper v

    def __cinit__(self):
        placement_new(self.v)
    def __dealloc__(self):
        placement_delete(self.v)

    def __call__(self, *args, **kwargs):
        return ndt_type_callable_call(GET(self.v), args, kwargs)

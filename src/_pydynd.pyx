#
# Copyright (C) 2011-13, DyND Developers
# BSD 2-Clause License, see LICENSE.txt
#

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
    object ndobject_as_numpy_struct_capsule(ndobject&) except +translate_exception
    void import_numpy()
import_numpy()

# Initialize ctypes C level interop data
cdef extern from "ctypes_interop.hpp" namespace "pydynd":
    void init_ctypes_interop() except +translate_exception
init_ctypes_interop()

# Initialize C++ access to the Cython type objects
init_w_ndobject_typeobject(w_ndobject)
init_w_dtype_typeobject(w_dtype)
init_w_ndobject_callable_typeobject(w_ndobject_callable)
init_w_dtype_callable_typeobject(w_dtype_callable)

include "dynd.pxd"
#include "codegen_cache.pxd"
include "dtype.pxd"
include "ndobject.pxd"
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
                str(which_diagnostics_enabled().c_str()), PerformanceWarning)

from cython.operator import dereference

# Create the codegen cache used by default when making gfuncs
#cdef w_codegen_cache default_cgcache_c = w_codegen_cache()
# Expose it outside the module too
#default_cgcache = default_cgcache_c

# Expose the git hashes and version numbers of this build
_dynd_version_string = str(dynd_version_string)
_dynd_git_sha1 = str(dynd_git_sha1)
_dynd_python_version_string = str(dynd_python_version_string)
_dynd_python_git_sha1 = str(dynd_python_git_sha1)

cdef class w_dtype:
    """
    dtype(obj=None)

    Create a dynd type object.

    A dynd type object describes the dimensional
    structure and element type of a dynd ndobject.

    Parameters
    ----------
    obj : string or other data type, optional
        A Blaze datashape string or a data type from another
        system such as NumPy or ctypes.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> nd.dtype('int16')
    ndt.int16
    >>> nd.dtype('5, VarDim, float32')
    nd.dtype('fixed_dim<5, var_dim<float32>>')
    >>> nd.dtype('{x: float32; y: float32; z: float32}')
    nd.dtype('fixedstruct<float32 x, float32 y, float32 z>')
    """
    # To access the embedded dtype, use "GET(self.v)",
    # which returns a reference to the dtype, and
    # SET(self.v, <dtype value>), which sets the embedded
    # dtype's value.
    cdef dtype_placement_wrapper v

    def __cinit__(self, rep=None):
        placement_new(self.v)
        if rep is not None:
            SET(self.v, make_dtype_from_pyobject(rep))
    def __dealloc__(self):
        placement_delete(self.v)

    def __dir__(self):
        # Customize dir() so that additional properties of various dtypes
        # will show up in IPython tab-complete, for example.
        result = dict(w_dtype.__dict__)
        result.update(object.__dict__)
        add_dtype_names_to_dir_dict(GET(self.v), result)
        return result.keys()

    def __call__(self, *args, **kwargs):
        return call_dtype_constructor_function(GET(self.v), args, kwargs)

    def __getattr__(self, name):
        return get_dtype_dynamic_property(GET(self.v), name)

    property data_size:
        """
        a.data_size

        The size, in bytes, of the data for an instance
        of this dynd type.
        """
        def __get__(self):
            return GET(self.v).get_data_size()

    property alignment:
        """
        a.alignment

        The alignment, in bytes, of the data for an
        instance of this dynd type.
        
        Data for this dynd type must always be aligned
        according to this alignment, unaligned data
        requires an adapter transformation applied.
        """
        def __get__(self):
            return GET(self.v).get_alignment()

    property kind:
        """
        a.kind

        The kind of this dynd type, as a string.

        Example kinds are 'bool', 'int', 'uint',
        'real', 'complex', 'string', 'uniform_array',
        'expression'.
        """
        def __get__(self):
            return dtype_get_kind(GET(self.v))

    property type_id:
        """
        a.type_id

        The type id of this dynd type, as a string.

        Example type ids are 'bool', 'int8', 'uint32',
        'float64', 'complex_float32', 'string', 'byteswap'.
        """
        def __get__(self):
            return dtype_get_type_id(GET(self.v))

    property undim:
        """
        a.undim

        The number of uniform dimensions in this dynd type.

        This property is roughly equivalent to NumPy
        ndarray's 'ndim'. Indexing with [] can in many cases
        go deeper than just the uniform dimensions, for
        example structs can be indexed this way.
        """
        def __get__(self):
            return GET(self.v).get_undim()

    property dshape:
        """
        a.dshape

        The Blaze datashape of the dynd type, as a string.
        """
        def __get__(self):
            return str(dynd_format_datashape(GET(self.v)).c_str())

    property udtype:
        """
        a.udtype

        The dynd type of the element after the 'undim'
        uniform dimensions are indexed away.

        This property is roughly equivalent to NumPy
        ndarray's 'dtype'.
        """
        def __get__(self):
            cdef w_dtype result = w_dtype()
            SET(result.v, GET(self.v).get_udtype())
            return result;

    property value_dtype:
        """
        a.value_dtype

        If this is an expression dynd type, returns the
        dynd type that values after evaluation have. Otherwise,
        returns this dynd type unchanged.
        """
        def __get__(self):
            cdef w_dtype result = w_dtype()
            SET(result.v, GET(self.v).value_dtype())
            return result

    property operand_dtype:
        """
        a.operand_dtype

        If this is an expression dynd type, returns the
        dynd type that inputs to its expression evaluation
        have. Otherwise, returns this dynd type unchanged.
        """
        def __get__(self):
            cdef w_dtype result = w_dtype()
            SET(result.v, GET(self.v).operand_dtype())
            return result

    property canonical_dtype:
        """
        a.canonical_dtype

        Returns a version of this dtype that is canonical,
        where any intermediate pointers are removed and expressions
        are stripped away.
        """
        def __get__(self):
            cdef w_dtype result = w_dtype()
            SET(result.v, GET(self.v).get_canonical_dtype())
            return result

    property property_names:
        """
        a.property_names

        Returns the names of properties exposed by ndobjects
        of this dtype.
        """
        def __get__(self):
            return dtype_ndobject_property_names(GET(self.v))

    def __getitem__(self, x):
        cdef w_dtype result = w_dtype()
        SET(result.v, dtype_getitem(GET(self.v), x))
        return result

    def __str__(self):
        return str(dtype_str(GET(self.v)).c_str())

    def __repr__(self):
        return str(dtype_repr(GET(self.v)).c_str())

    def __richcmp__(lhs, rhs, int op):
        if op == Py_EQ:
            if type(lhs) == w_dtype and type(rhs) == w_dtype:
                return GET((<w_dtype>lhs).v) == GET((<w_dtype>rhs).v)
            else:
                return False
        elif op == Py_NE:
            if type(lhs) == w_dtype and type(rhs) == w_dtype:
                return GET((<w_dtype>lhs).v) != GET((<w_dtype>rhs).v)
            else:
                return False
        return NotImplemented

def replace_udtype(w_dtype dt, replacement_dt, size_t replace_undim=0):
    """
    replace_udtype(dt, replacement_dt, replace_undim=0)

    Replaces the uniform dtype with the replacement.
    If `replace_undim` is positive, that number of uniform
    dimensions are replaced as well.

    Parameters
    ----------
    dt : dynd type
        The dtype whose uniform dtype is to be replaced.
    replacement_dt : dynd type
        The replacement dynd type.
    replace_undim : integer, optional
        If positive, this is the number of uniform
        dimensions which are replaced in addition to
        the uniform dtype.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> d = nd.dtype('3, VarDim, int32')
    >>> ndt.replace_udtype(d, 'M, float64')
    nd.dtype('fixed_dim<3, var_dim<strided_dim<float64>>>')
    >>> ndt.replace_udtype(d, '{x: int32; y:int32}', 1)
    nd.dtype('fixed_dim<3, fixedstruct<int32 x, int32 y>>')
    """
    cdef w_dtype result = w_dtype()
    SET(result.v, GET(dt.v).with_replaced_udtype(GET(w_dtype(replacement_dt).v), replace_undim))
    return result

def extract_udtype(dt, size_t keep_undim=0):
    """
    extract_udtype(dt, keep_undim=0)

    Extracts the uniform type from the provided
    dynd type. If `keep_undim` is positive, that
    many uniform dimensions are kept in the result.

    Parameters
    ----------
    dt : dynd type
        The dtype whose uniform dtype is to be extracted.
    keep_undim : integer, optional
        If positive, this is the number of uniform
        dimensions which are kept in addition to
        the uniform dtype.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> d = nd.dtype('3, VarDim, int32')
    >>> ndt.extract_udtype(d)
    ndt.int32
    >>> ndt.extract_udtype(d, 1)
    nd.dtype('var_dim<int32>')
    """
    cdef w_dtype result = w_dtype()
    SET(result.v, GET(w_dtype(dt).v).get_udtype(keep_undim))
    return result

def make_byteswap_dtype(builtin_dtype, operand_dtype=None):
    """
    make_byteswap_dtype(builtin_dtype, operand_dtype=None)

    Constructs a byteswap dtype from a builtin one, with an
    optional expression dtype to chain in as the operand.

    Parameters
    ----------
    builtin_dtype : dynd type
        The builtin dynd type (like ndt.int16, ndt.float64) to
        which to apply the byte swap operation.
    operand_dtype: dynd type, optional
        An expression dynd type whose value type is a fixed bytes
        dynd type with the same data size and alignment as
        'builtin_dtype'.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> ndt.make_byteswap_dtype(ndt.int16)
    nd.dtype('byteswap<int16>')
    """
    cdef w_dtype result = w_dtype()
    if operand_dtype is None:
        SET(result.v, dynd_make_byteswap_dtype(GET(w_dtype(builtin_dtype).v)))
    else:
        SET(result.v, dynd_make_byteswap_dtype(GET(w_dtype(builtin_dtype).v), GET(w_dtype(operand_dtype).v)))
    return result

def make_fixedbytes_dtype(int data_size, int data_alignment=1):
    """
    make_fixedbytes_dtype(data_size, data_alignment=1)

    Constructs a bytes dtype with the specified data size and alignment.

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

    >>> ndt.make_fixedbytes_dtype(4)
    nd.dtype('fixedbytes<4,1>')
    >>> ndt.make_fixedbytes_dtype(6, 2)
    nd.dtype('fixedbytes<6,2>')
    """
    cdef w_dtype result = w_dtype()
    SET(result.v, dynd_make_fixedbytes_dtype(data_size, data_alignment))
    return result

def make_convert_dtype(to_dtype, from_dtype, errmode=None):
    """
    make_convert_dtype(to_dtype, from_dtype, errmode='fractional')
    
    Constructs an expression dtype which converts from one
    dynd type to another, using a specified mode for handling
    conversion errors.

    Parameters
    ----------
    to_type : dynd type
        The dynd type being converted to. This is the 'value_dtype'
        of the resulting expression dynd type.
    from_type : dynd type
        The dynd type being converted from. This is the 'operand_dtype'
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

    >>> ndt.make_convert_dtype(ndt.int16, ndt.float32)
    nd.dtype('convert<to=int16, from=float32>')
    >>> ndt.make_convert_dtype(ndt.uint8, ndt.uint16, 'none')
    nd.dtype('convert<to=uint8, from=uint16, errmode=none>')
    """
    cdef w_dtype result = w_dtype()
    SET(result.v, dynd_make_convert_dtype(GET(w_dtype(to_dtype).v), GET(w_dtype(from_dtype).v), errmode))
    return result

def make_unaligned_dtype(aligned_dtype):
    """
    make_unaligned_dtype(aligned_dtype)

    Constructs a dtype with alignment of 1 from the given dtype.
    If the dtype already has alignment 1, just returns it.

    Parameters
    ----------
    aligned_dtype : dynd type
        The dynd type which should be viewed on data that is
        not properly aligned.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> ndt.make_unaligned_dtype(ndt.int32)
    nd.dtype('unaligned<int32>')
    >>> ndt.make_unaligned_dtype(ndt.uint8)
    ndt.uint8
    """
    cdef w_dtype result = w_dtype()
    SET(result.v, dynd_make_unaligned_dtype(GET(w_dtype(aligned_dtype).v)))
    return result

def make_fixedstring_dtype(int size, encoding=None):
    """
    make_fixedstring_dtype(size, encoding='utf_8')

    Constructs a fixed-size string dtype with a specified encoding,
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

    >>> ndt.make_fixedstring_dtype(10)
    nd.dtype('string<10>')
    >>> ndt.make_fixedstring_dtype(10, 'utf_32')
    nd.dtype('string<10,utf_32>')
    """
    cdef w_dtype result = w_dtype()
    SET(result.v, dynd_make_fixedstring_dtype(size, encoding))
    return result

def make_string_dtype(encoding=None):
    """
    make_string_dtype(encoding='utf_8')
    
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

    >>> ndt.make_string_dtype()
    ndt.string
    >>> ndt.make_string_dtype('utf_16')
    nd.dtype('string<utf_16>')
    """
    cdef w_dtype result = w_dtype()
    SET(result.v, dynd_make_string_dtype(encoding))
    return result

def make_pointer_dtype(target_dtype):
    """
    make_pointer_dtype(target_dtype)

    Constructs a dynd type which is a pointer to the target type.

    Parameters
    ----------
    target_dtype : dynd type
        The type that the pointer points to. This is similar to
        the '*' in C/C++ type declarations.
    """
    cdef w_dtype result = w_dtype()
    SET(result.v, dynd_make_pointer_dtype(GET(w_dtype(target_dtype).v)))
    return result

def make_strided_dim_dtype(element_dtype, undim=None):
    """
    make_strided_dim_dtype(element_dtype, undim=1)

    Constructs an array dynd type with one or more strided
    dimensions. A single strided_dim dynd type corresponds
    to one dimension, so when undim > 1, multiple strided_dim
    dimensions are created.

    Parameters
    ----------
    element_dtype : dynd type
        The type of one element in the strided array.
    undim : int
        The number of uniform strided_dim dimensions to create.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> ndt.make_strided_dim_dtype(ndt.int32)
    nd.dtype('strided_dim<int32>')
    >>> ndt.make_strided_dim_dtype(ndt.int32, 3)
    nd.dtype('strided_dim<strided_dim<strided_dim<int32>>>')
    """
    cdef w_dtype result = w_dtype()
    if (undim is None):
        SET(result.v, dynd_make_strided_dim_dtype(GET(w_dtype(element_dtype).v)))
    else:
        SET(result.v, dynd_make_strided_dim_dtype(GET(w_dtype(element_dtype).v), int(undim)))
    return result

def make_fixed_dim_dtype(shape, element_dtype, axis_perm=None):
    """
    make_fixed_dim_dtype(shape, element_dtype, axis_perm=None)
    
    Constructs a fixed_dim dtype of the given shape and axis permutation
    (default C order).

    Parameters
    ----------
    shape : tuple of int
        The multi-dimensional shape of the resulting fixed array dtype.
    element_dtype : dynd type
        The type of each element in the resulting array type.
    axis_perm : tuple of int
        If not provided, C-order is used. Must be a permutation of
        the integers 0 through len(shape)-1, ordered so each
        value increases with the size of the strides. [N-1, ..., 0]
        gives C-order, and [0, ..., N-1] gives F-order.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> ndt.make_fixed_dim_dtype(5, ndt.int32)
    nd.dtype('fixed_dim<5, int32>')
    >>> ndt.make_fixed_dim_dtype((3,5), ndt.int32)
    nd.dtype('fixed_dim<3, fixed_dim<5, int32>>')
    >>> ndt.make_fixed_dim_dtype((3,5), ndt.int32, axis_perm=(0,1))
    nd.dtype('fixed_dim<3, stride=4, fixed_dim<5, stride=12, int32>>')
    """
    cdef w_dtype result = w_dtype()
    SET(result.v, dynd_make_fixed_dim_dtype(shape, GET(w_dtype(element_dtype).v), axis_perm))
    return result

def make_fixedstruct_dtype(field_types, field_names):
    """
    make_fixedstruct_dtype(field_types, field_names)

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

    >>> ndt.make_fixedstruct_dtype([ndt.int32, ndt.float64], ['x', 'y'])
    nd.dtype('fixedstruct<int32 x, float64 y>')
    """
    cdef w_dtype result = w_dtype()
    SET(result.v, dynd_make_fixedstruct_dtype(field_types, field_names))
    return result

def make_struct_dtype(field_types, field_names):
    """
    make_struct_dtype(field_types, field_names)

    Constructs a struct dynd type, which has fields with a flexible
    per-ndobject layout.
    
    If a subset of fields from a fixed_struct are taken,
    the result is a struct, with the layout specified
    in the ndobject's metadata.

    Parameters
    ----------
    field_types : list of dynd types
        A list of types, one for each field.
    field_names : list of strings
        A list of names, one for each field, corresponding to 'field_types'.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> ndt.make_struct_dtype([ndt.int32, ndt.float64], ['x', 'y'])
    nd.dtype('struct<int32 x, float64 y>')
    """
    cdef w_dtype result = w_dtype()
    SET(result.v, dynd_make_struct_dtype(field_types, field_names))
    return result

def make_var_dim_dtype(element_dtype):
    """
    make_fixed_dim_dtype(element_dtype)
    
    Constructs a var_dim dtype.

    Parameters
    ----------
    element_dtype : dynd type
        The type of each element in the resulting array type.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> ndt.make_var_dim_dtype(ndt.float32)
    nd.dtype('var_dim<float32>')
    """
    cdef w_dtype result = w_dtype()
    SET(result.v, dynd_make_var_dim_dtype(GET(w_dtype(element_dtype).v)))
    return result

def make_categorical_dtype(values):
    """
    make_categorical_dtype(values)

    Constructs a categorical dynd type with the
    specified values as its categories.

    Instances of the resulting type are integers, consisting
    of indices into this values array. The size of the values
    array controls what kind of integers are used, if there
    are 256 or fewer categories, a uint8 is used, if 65536 or
    fewer, a uint16 is used, and otherwise a uint32 is used.

    Parameters
    ----------
    values : one-dimensional ndobject
        This is an array of the values that become the categories
        of the resulting type. The values must be unique.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> ndt.make_categorical_dtype(['sunny', 'rainy', 'cloudy', 'stormy'])
    nd.dtype('categorical<string<ascii>, ["sunny", "rainy", "cloudy", "stormy"]>')
    """
    cdef w_dtype result = w_dtype()
    SET(result.v, dynd_make_categorical_dtype(GET(w_ndobject(values).v)))
    return result

def factor_categorical_dtype(values):
    """
    factor_categorical_dtype(values)

    Constructs a categorical dynd type with the
    unique sorted subset of the values as its
    categories.

    Parameters
    ----------
    values : one-dimensional ndobject
        This is an array of the values that are sorted, with
        duplicates removed, to produce the categories of
        the resulting dynd type.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> ndt.factor_categorical_dtype(['M', 'M', 'F', 'F', 'M', 'F', 'M'])
    nd.dtype('categorical<string<ascii>, ["F", "M"]>')
    """
    cdef w_dtype result = w_dtype()
    SET(result.v, dynd_factor_categorical_dtype(GET(w_ndobject(values).v)))
    return result

##############################################################################

# NOTE: This is a possible alternative to the init_w_ndobject_typeobject() call
#       above, but it generates a 1300 line header file and still requires calling
#       import__dnd from the C++ code, so directly using C++ primitives seems simpler.
#cdef public api class w_ndobject [object WNDArrayObject, type WNDArrayObject_Type]:

cdef class w_ndobject:
    """
    ndobject(obj=None, udtype=None, dtype=None)

    Create a dynd ndobject out of the provided object.

    The ndobject is the dynamically typed multi-dimensional
    object provided by the dynd library. It is similar to
    NumPy's ndarray, but has its dimensional structure encoded
    in the dtype, along with the element type.

    When given a NumPy array, the resulting ndobject is a view
    into the NumPy array data. When given lists of Python object,
    an attempt is made to deduce an appropriate dynd type for
    the ndobject, and a conversion is made if possible, or an
    exception is raised.

    Parameters
    ----------
    obj : multi-dimensional object, optional
        Any object which dynd knows how to interpret as an ndobject.
    udtype: dynd type
        If provided, the type is used as the uniform type for the
        input, and the shape of the leading dimensions is deduced.
        This parameter cannot be used together with 'dtype'.
    dtype: dynd type
        If provided, the type is used as the full type for the input.
        If needed by the type, the shape is deduced from the input.
        This parameter cannot be used together with 'udtype'.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> nd.ndobject([1, 2, 3, 4, 5])
    nd.ndobject([1, 2, 3, 4, 5], strided_dim<int32>)
    >>> nd.ndobject([[1, 2], [3, 4, 5.0]])
    nd.ndobject([[1, 2], [3, 4, 5]], strided_dim<var_dim<float64>>)
    >>> from datetime import date
    >>> nd.ndobject([date(2000,2,14), date(2012,1,1)])
    nd.ndobject([2000-02-14, 2012-01-01], strided_dim<date>)
    """
    # To access the embedded dtype, use "GET(self.v)",
    # which returns a reference to the ndobject, and
    # SET(self.v, <ndobject value>), which sets the embeded
    # ndobject's value.
    cdef ndobject_placement_wrapper v

    def __cinit__(self, obj=None, udtype=None, dtype=None):
        placement_new(self.v)
        if obj is not None:
            # Get the array data
            if udtype is not None:
                if dtype is not None:
                    raise ValueError('Must provide only one of udtype or dtype, not both')
                ndobject_init_from_pyobject(GET(self.v), obj, udtype, True)
            elif dtype is not None:
                ndobject_init_from_pyobject(GET(self.v), obj, dtype, False)
            else:
                ndobject_init_from_pyobject(GET(self.v), obj)

    def __dealloc__(self):
        placement_delete(self.v)

    def __dir__(self):
        # Customize dir() so that additional properties of various dtypes
        # will show up in IPython tab-complete, for example.
        result = dict(w_ndobject.__dict__)
        result.update(object.__dict__)
        add_ndobject_names_to_dir_dict(GET(self.v), result)
        return result.keys()

    def __getattr__(self, name):
        return get_ndobject_dynamic_property(GET(self.v), name)

    def __setattr__(self, name, value):
        set_ndobject_dynamic_property(GET(self.v), name, value)

    def __contains__(self, x):
        return ndobject_contains(GET(self.v), x)

    def debug_repr(self):
        """
        a.debug_repr()

        Returns a raw representation of the ndobject data.

        This can be useful for diagnosing bugs in the ndobject
        or dtype/metadata/data abstraction ndobjects are based on.
        """
        return str(ndobject_debug_print(GET(self.v)).c_str())

    def eval(self):
        """
        a.eval()
        
        Returns a version of the ndobject with plain values,
        all expressions evaluated. This returns the original
        array back if it has no expression type.

        Examples
        --------
        >>> from dynd import nd, ndt

        >>> a = nd.ndobject([1.5, 2, 3])
        >>> a
        nd.ndobject([1.5, 2, 3], strided_dim<float64>)
        >>> b = a.ucast(ndt.int16, errmode='none')
        >>> b
        nd.ndobject([1, 2, 3], strided_dim<convert<to=int16, from=float64, errmode=none>>)
        >>> b.eval()
        nd.ndobject([1, 2, 3], strided_dim<int16>)
        """
        cdef w_ndobject result = w_ndobject()
        SET(result.v, ndobject_eval(GET(self.v)))
        return result

    def eval_immutable(self):
        """
        a.eval_immutable()

        Evaluates into an immutable ndobject. If the ndobject is
        already immutable and not an expression type, returns it
        as is.
        """
        cdef w_ndobject result = w_ndobject()
        SET(result.v, GET(self.v).eval_immutable())
        return result

    def eval_copy(self, access=None):
        """
        a.eval_copy(access='readwrite')

        Evaluates into a new ndobject, guaranteeing a copy is made.

        Parameters
        ----------
        access : 'readwrite' or 'immutable'
            Specifies the access control of the resulting copy.
        """
        cdef w_ndobject result = w_ndobject()
        SET(result.v, ndobject_eval_copy(GET(self.v), access))
        return result

    def storage(self):
        """
        a.storage()

        Returns a version of the ndobject with its storage dtype,
        all expressions discarded. For data types that are plain
        old data, views them as a bytes dtype.

        Examples
        --------
        >>> from dynd import nd, ndt

        >>> a = nd.ndobject([1, 2, 3], dtype=ndt.int16)
        >>> a
        nd.ndobject([1, 2, 3], strided_dim<int16>)
        >>> a.storage()
        nd.ndobject([0x0100, 0x0200, 0x0300], strided_dim<fixedbytes<2,2>>)
        """
        cdef w_ndobject result = w_ndobject()
        SET(result.v, GET(self.v).storage())
        return result

    def ucast(self, dtype, int replace_undim=0, errmode=None):
        """
        a.ucast(dtype, replace_undim=0, errmode='fractional')
        
        Casts the ndobject's uniform dtype to the requested dtype,
        producing a conversion dtype. The uniform dtype is the dtype
        after the a.undim uniform dimensions.

        Parameters
        ----------
        dtype : dynd type
            The uniform dtype is cast into this type.
            If `replace_undim` is not zero, then that many
            dimensions are included in what is cast as well.
        replace_undim : integer, optional
            The number of uniform dimensions to replace in doing
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
        >>> a = nd.ndobject([date(1929,3,13), date(1979,3,22)]).ucast('{month: int32; year: int32; day: float32}')
        >>> a
        nd.ndobject([[3, 1929, 13], [3, 1979, 22]], strided_dim<convert<to=fixedstruct<int32 month, int32 year, float32 day>, from=date>>)
        >>> a.eval()
        nd.ndobject([[3, 1929, 13], [3, 1979, 22]], strided_dim<fixedstruct<int32 month, int32 year, float32 day>>)
        """
        cdef w_ndobject result = w_ndobject()
        SET(result.v, ndobject_ucast(GET(self.v), GET(w_dtype(dtype).v), replace_undim, errmode))
        return result

    def view_scalars(self, dtype):
        """
        a.view_scalars(dtype)
        
        Views the data of the ndobject as the requested dtype,
        where it makes sense.

        If the ndobject is a one-dimensional contiguous
        array of plain old data, the new dtype may have a different
        element size than original one.

        When the ndobject has an expression type, the
        view is created by layering another view dtype
        onto the ndobject's existing expression.

        Parameters
        ----------
        dtype : dynd type
            The scalars are viewed as this dtype.
        """
        cdef w_ndobject result = w_ndobject()
        SET(result.v, GET(self.v).view_scalars(GET(w_dtype(dtype).v)))
        return result

    def flag_as_immutable(self):
        """
        a.flag_as_immutable()
        
        When there's still only one reference to an
        ndobject, can be used to flag it as immutable.
        """
        GET(self.v).flag_as_immutable()

    property dtype:
        """
        a.dtype

        The dynd type of the ndobject. This is the full
        data type, including its multi-dimensional structure.

        Examples
        --------
        >>> from dynd import nd, ndt

        >>> nd.ndobject([1,2,3,4]).dtype
        nd.dtype('strided_dim<int32>')
        >>> nd.ndobject([[1,2],[3.0]]).dtype
        nd.dtype('strided_dim<var_dim<float64>>')
        """
        def __get__(self):
            cdef w_dtype result = w_dtype()
            SET(result.v, GET(self.v).get_dtype())
            return result

    property dshape:
        """
        a.dshape

        The Blaze datashape of the ndobject, as a string.
        """
        def __get__(self):
            return str(dynd_format_datashape(GET(self.v)).c_str())

    property udtype:
        """
        a.udtype

        The uniform dynd type of the ndobject. This is
        the type after removing all the a.undim uniform
        dimensions from a.dtype.
        """
        def __get__(self):
            cdef w_dtype result = w_dtype()
            SET(result.v, GET(self.v).get_dtype().get_udtype())
            return result

    property undim:
        """
        a.undim

        The number of uniform dimensions in the ndobject.
        This roughly corresponds to the number of dimensions
        in a NumPy array.
        """
        def __get__(self):
            return GET(self.v).get_dtype().get_undim()

    property is_scalar:
        """
        a.is_scalar

        True if the ndobject is a scalar.
        """
        def __get__(self):
            return GET(self.v).is_scalar()

    property shape:
        def __get__(self):
            return ndobject_get_shape(GET(self.v))

    property strides:
        def __get__(self):
            return ndobject_get_strides(GET(self.v))

    def __str__(self):
        return ndobject_str(GET(self.v))

    def __unicode__(self):
        return ndobject_unicode(GET(self.v))

    def __index__(self):
        return ndobject_index(GET(self.v))

    def __nonzero__(self):
        return ndobject_nonzero(GET(self.v))

    def __repr__(self):
        return str(ndobject_repr(GET(self.v)).c_str())

    def __len__(self):
        if GET(self.v).is_scalar():
            raise TypeError('zero-dimensional dynd::ndobject has no len()')
        return GET(self.v).get_dim_size()

    def __getitem__(self, x):
        cdef w_ndobject result = w_ndobject()
        SET(result.v, ndobject_getitem(GET(self.v), x))
        return result

    def __setitem__(self, x, y):
        ndobject_setitem(GET(self.v), x, y)

    def __getbuffer__(w_ndobject self, Py_buffer* buffer, int flags):
        # Docstring triggered Cython bug (fixed in master), so it's commented out
        #"""PEP 3118 buffer protocol"""
        ndobject_getbuffer_pep3118(self, buffer, flags)

    def __releasebuffer__(w_ndobject self, Py_buffer* buffer):
        # Docstring triggered Cython bug (fixed in master), so it's commented out
        #"""PEP 3118 buffer protocol"""
        ndobject_releasebuffer_pep3118(self, buffer)

    def __add__(lhs, rhs):
        cdef w_ndobject result = w_ndobject()
        SET(result.v, ndobject_add(GET(w_ndobject(lhs).v), GET(w_ndobject(rhs).v)))
        return result

    def __sub__(lhs, rhs):
        cdef w_ndobject result = w_ndobject()
        SET(result.v, ndobject_subtract(GET(w_ndobject(lhs).v), GET(w_ndobject(rhs).v)))
        return result

    def __mul__(lhs, rhs):
        cdef w_ndobject result = w_ndobject()
        SET(result.v, ndobject_multiply(GET(w_ndobject(lhs).v), GET(w_ndobject(rhs).v)))
        return result

    def __div__(lhs, rhs):
        cdef w_ndobject result = w_ndobject()
        SET(result.v, ndobject_divide(GET(w_ndobject(lhs).v), GET(w_ndobject(rhs).v)))
        return result

def as_py(w_ndobject n):
    """
    nd.as_py(n)

    Evaluates the dynd ndobject, converting it into native Python types.

    Uniform dimensions convert into Python lists, struct types convert
    into Python dicts, scalars convert into the most appropriate Python
    scalar for their type.

    Parameters
    ----------
    n : dynd ndobject
        The ndobject to convert into native Python types.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> a = nd.ndobject([1, 2, 3, 4.0])
    >>> a
    nd.ndobject([1, 2, 3, 4], strided_dim<float64>)
    >>> nd.as_py(a)
    [1.0, 2.0, 3.0, 4.0]
    """
    return ndobject_as_py(GET(n.v))

def as_numpy(w_ndobject n, allow_copy=False):
    """
    nd.as_numpy(n, allow_copy=False)
    
    Evaluates the dynd ndobject, and converts it into a NumPy object.
    
    Parameters
    ----------
    n : dynd ndobject
        The ndobject to convert into native Python types.
    allow_copy : bool, optional
        If true, allows a copy to be made when the ndobject types
        can't be directly viewed as a NumPy array, but with a
        data-preserving copy they can be.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> import numpy as np
    >>> a = nd.ndobject([[1, 2, 3], [4, 5, 6]])
    >>> a
    nd.ndobject([[1, 2, 3], [4, 5, 6]], strided_dim<strided_dim<int32>>)
    >>> nd.as_numpy(a)
    array([[1, 2, 3],
           [4, 5, 6]])
    """
    # TODO: Could also convert dynd dtypes into numpy dtypes
    return ndobject_as_numpy(n, bool(allow_copy))

def empty(shape, dtype=None):
    """
    nd.empty(dtype)
    nd.empty(shape, dtype)
    
    Creates an uninitialized ndobject of the specified
    (shape, dtype) or just (dtype).
    
    Parameters
    ----------
    shape : list of int, optional
        If provided, specifies the shape for the dtype dimensions
        that don't encode a dimension size themselves, such as
        strided_dim dimensions.
    dtype : dynd type
        The data type of the uninitialized array to create. This
        is the full data type, including the multi-dimensional
        structure.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> nd.empty('2, 2, int8')
    nd.ndobject([[0, -24], [0, 4]], fixed_dim<2, fixed_dim<2, int8>>)
    >>> nd.empty((2, 2), 'M, N, int16')
    nd.ndobject([[179, 0], [0, 16816]], strided_dim<strided_dim<int16>>)
    """
    cdef w_ndobject result = w_ndobject()
    if dtype is not None:
        SET(result.v, ndobject_empty(shape, GET(w_dtype(dtype).v)))
    else:
        # Interpret the first argument (shape) as a dtype in the one argument case
        SET(result.v, ndobject_empty(GET(w_dtype(shape).v)))
    return result

def empty_like(w_ndobject prototype, dtype=None):
    """
    nd.empty_like(prototype, dtype=None)

    Creates an uninitialized ndobject whose uniform dimensions match
    the structure of the prototype's uniform dimensions.
    
    Parameters
    ----------
    prototype : dynd ndobject
        The ndobject whose structure is to be matched.
    dtype : dynd type, optional
        If provided, replaces the prototype's uniform dtype in
        the result.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> a = nd.ndobject([[1, 2], [3, 4]])
    >>> a
    nd.ndobject([[1, 2], [3, 4]], strided_dim<strided_dim<int32>>)
    >>> nd.empty_like(a)
    nd.ndobject([[808529973, 741351468], [0, 0]], strided_dim<strided_dim<int32>>)
    >>> nd.empty_like(a, dtype=ndt.float32)
    nd.ndobject([[1.47949e-041, 0], [0, 0]], strided_dim<strided_dim<float32>>)
    """
    cdef w_ndobject result = w_ndobject()
    if dtype is None:
        SET(result.v, ndobject_empty_like(GET(prototype.v)))
    else:
        SET(result.v, ndobject_empty_like(GET(prototype.v), GET(w_dtype(dtype).v)))
    return result

def groupby(data, by, groups = None):
    """
    nd.groupby(data, by, groups=None)
    
    Produces an array containing the elements of `data`, grouped
    according to `by` which has corresponding shape.
    
    Parameters
    ----------
    data : dynd ndobject
        A one-dimensional array of the data to be copied into
        the resulting grouped array.
    by : dynd ndobject
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
    nd.ndobject(["F", "M"], strided_dim<string<ascii>>)
    >>> a.eval()
    nd.ndobject([[2, 5, 6], [1, 3, 4]], fixed_dim<2, var_dim<int32>>)

    >>> a = nd.groupby([1, 2, 3, 4, 5, 6], ['M', 'F', 'M', 'M', 'F', 'F'], ['M', 'N', 'F'])
    >>> a.groups
    nd.ndobject(["M", "N", "F"], strided_dim<string<ascii>>)
    >>> a.eval()
    nd.ndobject([[1, 3, 4], [], [2, 5, 6]], fixed_dim<3, var_dim<int32>>)
    """
    cdef w_ndobject result = w_ndobject()
    if groups is None:
        SET(result.v, dynd_groupby(GET(w_ndobject(data).v), GET(w_ndobject(by).v)))
    else:
        if type(groups) in [list, w_ndobject]:
            # If groups is a list or ndobject, assume it's a list
            # of groups for a categorical dtype
            SET(result.v, dynd_groupby(GET(w_ndobject(data).v), GET(w_ndobject(by).v),
                            dynd_make_categorical_dtype(GET(w_ndobject(groups).v))))
        else:
            SET(result.v, dynd_groupby(GET(w_ndobject(data).v), GET(w_ndobject(by).v), GET(w_dtype(groups).v)))
    return result

def arange(start=None, stop=None, step=None, dtype=None):
    """
    nd.arange(stop, dtype=None)
    nd.arange(start, stop, step=None, dtype=None)
    
    Constructs an ndobject representing a stepped range of values.

    This function assumes that (stop-start)/step is approximately
    an integer, so as to be able to produce reasonable values with
    fractional steps which can't be exactly represented, such as 0.1.
    
    Parameters
    ----------
    start : int, optional
        If provided, this is the first value in the resulting ndobject.
    stop : int
        This provides the stopping criteria for the range, and is
        not included in the resulting ndobject.
    step : int
        This is the increment.
    dtype : dynd type, optional
        If provided, it must be a scalar type, and the result
        is of this type.
    """
    cdef w_ndobject result = w_ndobject()
    # Move the first argument to 'stop' if stop isn't specified
    if stop is None:
        if start is not None:
            SET(result.v, ndobject_arange(None, start, step, dtype))
        else:
            raise ValueError("No value provided for 'stop'")
    else:
        SET(result.v, ndobject_arange(start, stop, step, dtype))
    return result

def linspace(start, stop, count=50, dtype=None):
    """
    nd.linspace(start, stop, count=50, dtype=None)
    
    Constructs a specified count of values interpolating a range.
    
    Parameters
    ----------
    start : floating point scalar
        The value of the first element of the resulting ndobject.
    stop : floating point scalar
        The value of the last element of the resulting ndobject.
    count : int, optional
        The number of elements in the resulting ndobject.
    dtype : dynd type, optional
        If provided, it must be a scalar type, and the result
        is of this type.
    """
    cdef w_ndobject result = w_ndobject()
    SET(result.v, ndobject_linspace(start, stop, count, dtype))
    return result

def fields(w_ndobject struct_array, *fields_list):
    """
    nd.fields(struct_array, *fields_list)

    Selects fields from an array of structs.

    Parameters
    ----------
    struct_array : dynd ndobject with struct udtype
        An ndobject whose uniform dtype has kind 'struct'. This
        could be a single struct instance, or an array of structs.
    *fields_list : string
        The remaining parameters must all be strings, and are the field
        names to select.
    """
    cdef w_ndobject result = w_ndobject()
    SET(result.v, nd_fields(GET(struct_array.v), fields_list))
    return result

def parse_json(dtype, json):
    """
    nd.parse_json(dtype, json)
    
    Parses an input JSON string as a particular dtype.

    Parameters
    ----------
    dtype : dynd type
        The type to interpret the input JSON as. If the data
        does not match this type, an error is raised during parsing.
    json : string or bytes
        String that contains the JSON to parse.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> nd.parse_json('VarDim, int8', '[1, 2, 3, 4, 5]')
    nd.ndobject([1, 2, 3, 4, 5], var_dim<int8>)
    >>> nd.parse_json('4, int8', '[1, 2, 3, 4]')
    nd.ndobject([1, 2, 3, 4], fixed_dim<4, int8>)
    >>> nd.parse_json('2, {x: int8; y: int8}', '[{"x":0, "y":1}, {"y":2, "x":3}]')
    nd.ndobject([[0, 1], [3, 2]], fixed_dim<2, fixedstruct<int8 x, int8 y>>)
    """
    cdef w_ndobject result = w_ndobject()
    if type(dtype) is w_ndobject:
        dynd_parse_json_ndobject(GET((<w_ndobject>dtype).v), GET(w_ndobject(json).v))
    else:
        SET(result.v, dynd_parse_json_dtype(GET(w_dtype(dtype).v), GET(w_ndobject(json).v)))
        return result

def format_json(w_ndobject n):
    """
    nd.format_json(n)
    
    Formats an ndobject as JSON.

    Parameters
    ----------
    n : dynd ndobject
        The object to format as JSON.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> a = nd.ndobject([[1, 2, 3], [1, 2]])
    >>> a
    nd.ndobject([[1, 2, 3], [1, 2]], strided_dim<var_dim<int32>>)
    >>> nd.format_json(a)
    nd.ndobject("[[1,2,3],[1,2]]", string)
    """
    cdef w_ndobject result = w_ndobject()
    SET(result.v, dynd_format_json(GET(n.v)))
    return result

def elwise_map(n, callable, dst_type, src_type = None):
    """
    nd.elwise_map(n, callable, dst_type, src_type=None)

    Applies a deferred element-wise mapping function to
    a dynd ndobject 'n'.

    Parameters
    ----------
    n : list of dynd ndobject
        A list of objects to which the mapping is applied.
    callable : Python callable
        A Python function which is called as
        'callable(dst, src[0], ..., src[N-1])', with
        one-dimensional dynd ndobjects 'dst', 'src[0]',
        ..., 'src[N-1]'. The function should, in an element-wise
        fashion, use the values in the different 'src'
        ndobjects to calculate values that get placed into
        the 'dst' ndobject. The function must not return a value.
    dst_type : dynd type
        The type of the computation's result.
    src_type : list of dynd type, optional
        A list of types of the source. If a source ndobject has
        a different type than the one corresponding in this list,
        it will be converted.
    """
    return dynd_elwise_map(n, callable, dst_type, src_type)

cdef class w_elwise_gfunc:
    cdef elwise_gfunc_placement_wrapper v

    def __cinit__(self, bytes name):
        placement_new(self.v, name)
    def __dealloc__(self):
        placement_delete(self.v)

    property name:
        def __get__(self):
            return str(GET(self.v).get_name().c_str())

#    def add_kernel(self, kernel, w_codegen_cache cgcache = default_cgcache_c):
#        """Adds a kernel to the gfunc object. Currently, this means a ctypes object with prototype."""
#        elwise_gfunc_add_kernel(GET(self.v), GET(cgcache.v), kernel)

    def debug_repr(self):
        """Prints a raw representation of the gfunc data."""
        return str(elwise_gfunc_debug_print(GET(self.v)).c_str())

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
            return str(GET(self.v).get_name().c_str())

#    def add_kernel(self, kernel, bint associative, bint commutative, identity = None, w_codegen_cache cgcache = default_cgcache_c):
#        """Adds a kernel to the gfunc object. Currently, this means a ctypes object with prototype."""
#        cdef w_ndobject id
#        if identity is None:
#            elwise_reduce_gfunc_add_kernel(GET(self.v), GET(cgcache.v), kernel, associative, commutative, ndobject())
#        else:
#            id = w_ndobject(identity)
#            elwise_reduce_gfunc_add_kernel(GET(self.v), GET(cgcache.v), kernel, associative, commutative, GET(id.v))

    def debug_repr(self):
        """Returns a raw representation of the gfunc data."""
        return str(elwise_reduce_gfunc_debug_print(GET(self.v)).c_str())

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
#        return str(codegen_cache_debug_print(GET(self.v)).c_str())

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

    def debug_repr(self):
        """Returns a raw representation of the elwise_program data."""
        return str(vm_elwise_program_debug_print(GET(self.v)).c_str())

cdef class w_ndobject_callable:
    cdef ndobject_callable_placement_wrapper v

    def __cinit__(self):
        placement_new(self.v)
    def __dealloc__(self):
        placement_delete(self.v)

    def __call__(self, *args, **kwargs):
        return ndobject_callable_call(GET(self.v), args, kwargs)

cdef class w_dtype_callable:
    cdef dtype_callable_placement_wrapper v

    def __cinit__(self):
        placement_new(self.v)
    def __dealloc__(self):
        placement_delete(self.v)

    def __call__(self, *args, **kwargs):
        return dtype_callable_call(GET(self.v), args, kwargs)

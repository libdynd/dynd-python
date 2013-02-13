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
    nd.dtype('fixedarray<5, var_array<float32>>')
    >>> nd.dtype('{x: float32; y: float32; z: float32}')
    nd.dtype('fixedstruct<float32 x, float32 y, float32 z>')
    """
    # To access the embedded dtype, use "GET(self.v)",
    # which returns a reference to the dtype, and
    # SET(self.v, <dtype value>), which sets the embeded
    # dtype's value.
    cdef dtype_placement_wrapper v

    def __cinit__(self, rep=None):
        placement_new(self.v)
        init_w_dtype_typeobject(type(self))
        if rep is not None:
            SET(self.v, make_dtype_from_object(rep))
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
        The size, in bytes, of the data for an instance
        of this dynd type.
        """
        def __get__(self):
            return GET(self.v).get_data_size()

    property alignment:
        """
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
        The kind of this dynd type, as a string.

        Example kinds are 'bool', 'int', 'uint',
        'real', 'complex', 'string', 'uniform_array',
        'expression'.
        """
        def __get__(self):
            return dtype_get_kind(GET(self.v))

    property type_id:
        """
        The type id of this dynd type, as a string.

        Example type ids are 'bool', 'int8', 'uint32',
        'float64', 'complex_float32', 'string', 'byteswap'.
        """
        def __get__(self):
            return dtype_get_type_id(GET(self.v))

    property undim:
        """
        The number of uniform dimensions in this dynd type.

        This property is roughly equivalent to NumPy
        ndarray's 'ndim'. Indexing with [] can in many cases
        go deeper than just the uniform dimensions, for
        example structs can be indexed this way.
        """
        def __get__(self):
            return GET(self.v).get_undim()

    property udtype:
        """
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
        Returns a version of this dtype that is canonical,
        where any intermediate pointers are removed and expressions
        are stripped away.
        """
        def __get__(self):
            cdef w_dtype result = w_dtype()
            SET(result.v, GET(self.v).get_canonical_dtype())
            return result

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
        SET(result.v, dnd_make_byteswap_dtype(GET(w_dtype(builtin_dtype).v)))
    else:
        SET(result.v, dnd_make_byteswap_dtype(GET(w_dtype(builtin_dtype).v), GET(w_dtype(operand_dtype).v)))
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
    SET(result.v, dnd_make_fixedbytes_dtype(data_size, data_alignment))
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
    SET(result.v, dnd_make_convert_dtype(GET(w_dtype(to_dtype).v), GET(w_dtype(from_dtype).v), errmode))
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
    SET(result.v, dnd_make_unaligned_dtype(GET(w_dtype(aligned_dtype).v)))
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
    SET(result.v, dnd_make_fixedstring_dtype(size, encoding))
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
    SET(result.v, dnd_make_string_dtype(encoding))
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
    SET(result.v, dnd_make_pointer_dtype(GET(w_dtype(target_dtype).v)))
    return result

def make_strided_array_dtype(element_dtype, undim=None):
    """
    make_strided_array_dtype(element_dtype, undim=1)

    Constructs an array dynd type with one or more strided
    dimensions. A single strided_array dynd type corresponds
    to one dimension, so when undim > 1, multipled strided_array
    dimensions are created.

    Parameters
    ----------
    element_dtype : dynd type
        The type of one element in the strided array.
    undim : int
        The number of uniform strided_array dimensions to create.

    Examples
    --------
    >>> from dynd import nd, ndt

    >>> ndt.make_strided_array_dtype(ndt.int32)
    nd.dtype('strided_array<int32>')
    >>> ndt.make_strided_array_dtype(ndt.int32, 3)
    nd.dtype('strided_array<strided_array<strided_array<int32>>>')
    """
    cdef w_dtype result = w_dtype()
    if (undim is None):
        SET(result.v, dnd_make_strided_array_dtype(GET(w_dtype(element_dtype).v)))
    else:
        SET(result.v, dnd_make_strided_array_dtype(GET(w_dtype(element_dtype).v), int(undim)))
    return result

def make_fixedarray_dtype(shape, element_dtype, axis_perm=None):
    """
    make_fixedarray_dtype(shape, element_dtype, axis_perm=None)
    
    Constructs a fixedarray dtype of the given shape and axis permutation
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

    >>> ndt.make_fixedarray_dtype(5, ndt.int32)
    nd.dtype('fixedarray<5, int32>')
    >>> ndt.make_fixedarray_dtype((3,5), ndt.int32)
    nd.dtype('fixedarray<3, fixedarray<5, int32>>')
    >>> ndt.make_fixedarray_dtype((3,5), ndt.int32, axis_perm=(0,1))
    nd.dtype('fixedarray<3, stride=4, fixedarray<5, stride=12, int32>>')
    """
    cdef w_dtype result = w_dtype()
    SET(result.v, dnd_make_fixedarray_dtype(shape, GET(w_dtype(element_dtype).v), axis_perm))
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
    SET(result.v, dnd_make_fixedstruct_dtype(field_types, field_names))
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
    SET(result.v, dnd_make_struct_dtype(field_types, field_names))
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
    SET(result.v, dnd_make_categorical_dtype(GET(w_ndobject(values).v)))
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
    SET(result.v, dnd_factor_categorical_dtype(GET(w_ndobject(values).v)))
    return result

##############################################################################

# NOTE: This is a possible alternative to the init_w_ndobject_typeobject() call
#       above, but it generates a 1300 line header file and still requires calling
#       import__dnd from the C++ code, so directly using C++ primitives seems simpler.
#cdef public api class w_ndobject [object WNDArrayObject, type WNDArrayObject_Type]:

cdef class w_ndobject:
    # To access the embedded dtype, use "GET(self.v)",
    # which returns a reference to the ndobject, and
    # SET(self.v, <ndobject value>), which sets the embeded
    # ndobject's value.
    cdef ndobject_placement_wrapper v

    def __cinit__(self, obj=None, dtype=None):
        placement_new(self.v)
        if obj is not None:
            # Get the array data
            ndobject_init_from_pyobject(GET(self.v), obj)

            # If a specific dtype is requested, use cast_scalars to switch types
            if dtype is not None:
                SET(self.v, GET(self.v).cast_scalars(GET(w_dtype(dtype).v), assign_error_default).vals())
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

    def debug_repr(self):
        """Returns a raw representation of the ndobject data."""
        return str(ndobject_debug_print(GET(self.v)).c_str())

    def eval(self):
        """Returns a version of the ndobject with plain values, all expressions evaluated."""
        cdef w_ndobject result = w_ndobject()
        SET(result.v, ndobject_vals(GET(self.v)))
        return result

    def eval_immutable(self):
        cdef w_ndobject result = w_ndobject()
        SET(result.v, GET(self.v).eval_immutable())
        return result

    def eval_copy(self, access_flags = None):
        cdef w_ndobject result = w_ndobject()
        SET(result.v, ndobject_eval_copy(GET(self.v), access_flags))
        return result

    def storage(self):
        """Returns a version of the ndobject with its storage dtype, all expressions discarded."""
        cdef w_ndobject result = w_ndobject()
        SET(result.v, GET(self.v).storage())
        return result

    def val_assign(self, obj):
        """Assigns to the ndobject by value instead of by reference."""
        cdef w_ndobject n = w_ndobject(obj)
        GET(self.v).val_assign(GET(n.v), assign_error_default)

    def cast_scalars(self, dtype, errmode=None):
        """Converts the ndobject's scalars to the requested dtype, producing a conversion dtype."""
        cdef w_ndobject result = w_ndobject()
        SET(result.v, ndobject_cast_scalars(GET(self.v), GET(w_dtype(dtype).v), errmode))
        return result

    def cast_udtype(self, dtype, errmode=None):
        """Converts the ndobject's uniform dtype (after all the uniform dimensions) to the requested dtype."""
        cdef w_ndobject result = w_ndobject()
        SET(result.v, ndobject_cast_udtype(GET(self.v), GET(w_dtype(dtype).v), errmode))
        return result

    def view_scalars(self, dtype):
        """Views the data of the ndobject as the requested dtype, where it makes sense."""
        cdef w_ndobject result = w_ndobject()
        SET(result.v, GET(self.v).view_scalars(GET(w_dtype(dtype).v)))
        return result

    def flag_as_immutable(self):
        """When there's still only one reference to an ndobject, can be used to flag it as immutable."""
        GET(self.v).flag_as_immutable()

    property dtype:
        def __get__(self):
            cdef w_dtype result = w_dtype()
            SET(result.v, GET(self.v).get_dtype())
            return result

    property dshape:
        def __get__(self):
            return str(dynd_format_datashape(GET(self.v)))

    property udtype:
        def __get__(self):
            cdef w_dtype result = w_dtype()
            SET(result.v, GET(self.v).get_dtype().get_udtype())
            return result

    property undim:
        def __get__(self):
            return GET(self.v).get_dtype().get_undim()

    property is_scalar:
        def __get__(self):
            return GET(self.v).is_scalar()

    property shape:
        def __get__(self):
            return ndobject_get_shape(GET(self.v))

    property strides:
        def __get__(self):
            return ndobject_get_strides(GET(self.v))

    def __str__(self):
        return str(ndobject_str(GET(self.v)).c_str())

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

def as_py(w_ndobject rhs):
    """Evaluates the dynd ndobject, and converts it into native Python types."""
    return ndobject_as_py(GET(rhs.v))

def as_numpy(w_ndobject rhs, allow_copy=False):
    """Evaluates the dynd ndobject, and converts it into a NumPy object."""
    # TODO: Could also convert dynd dtypes into numpy dtypes
    return ndobject_as_numpy(rhs, bool(allow_copy))

def empty(shape, dtype=None):
    """Creates an uninitialized array of the specified (shape, dtype) or just (dtype)."""
    cdef w_ndobject result = w_ndobject()
    if dtype is not None:
        SET(result.v, ndobject_empty(shape, GET(w_dtype(dtype).v)))
    else:
        # Interpret the first argument (shape) as a dtype in the one argument case
        SET(result.v, ndobject_empty(GET(w_dtype(shape).v)))
    return result

def empty_like(w_ndobject rhs, dtype=None):
    cdef w_ndobject result = w_ndobject()
    if dtype is None:
        SET(result.v, ndobject_empty_like(GET(rhs.v)))
    else:
        SET(result.v, ndobject_empty_like(GET(rhs.v), GET(w_dtype(dtype).v)))
    return result

def groupby(data, by, groups = None):
    """Produces an array containing the elements of `data`, grouped according to `by` which has corresponding shape."""
    cdef w_ndobject result = w_ndobject()
    if groups is None:
        SET(result.v, dynd_groupby(GET(w_ndobject(data).v), GET(w_ndobject(by).v)))
    else:
        if type(groups) in [list, w_ndobject]:
            # If groups is a list or ndobject, assume it's a list
            # of groups for a categorical dtype
            SET(result.v, dynd_groupby(GET(w_ndobject(data).v), GET(w_ndobject(by).v),
                            dnd_make_categorical_dtype(GET(w_ndobject(groups).v))))
        else:
            SET(result.v, dynd_groupby(GET(w_ndobject(data).v), GET(w_ndobject(by).v), GET(w_dtype(groups).v)))
    return result

def arange(start, stop=None, step=None):
    """Constructs an ndobject representing a stepped range of values."""
    cdef w_ndobject result = w_ndobject()
    # Move the first argument to 'stop' if stop isn't specified
    if stop is None:
        SET(result.v, ndobject_arange(None, start, step))
    else:
        SET(result.v, ndobject_arange(start, stop, step))
    return result

def linspace(start, stop, count=50):
    """Constructs a specified count of values interpolating a range."""
    cdef w_ndobject result = w_ndobject()
    SET(result.v, ndobject_linspace(start, stop, count))
    return result

def parse_json(dtype, json):
    """Parses an input JSON string as a particular dtype."""
    cdef w_ndobject result = w_ndobject()
    if type(dtype) is w_ndobject:
        dynd_parse_json_ndobject(GET((<w_ndobject>dtype).v), GET(w_ndobject(json).v))
    else:
        SET(result.v, dynd_parse_json_dtype(GET(w_dtype(dtype).v), GET(w_ndobject(json).v)))
        return result

def format_json(w_ndobject n):
    """Formats an ndobject as JSON."""
    cdef w_ndobject result = w_ndobject()
    SET(result.v, dynd_format_json(GET(n.v)))
    return result

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
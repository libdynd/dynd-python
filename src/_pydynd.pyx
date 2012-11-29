#
# Copyright (C) 2011-12, Dynamic NDArray Developers
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

include "dynd.pxd"
include "codegen_cache.pxd"
include "dtype.pxd"
include "ndobject.pxd"
include "elwise_gfunc.pxd"
include "elwise_reduce_gfunc.pxd"
include "vm_elwise_program.pxd"

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
cdef w_codegen_cache default_cgcache_c = w_codegen_cache()
# Expose it outside the module too
default_cgcache = default_cgcache_c

cdef class w_dtype:
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

    property element_size:
        def __get__(self):
            return GET(self.v).element_size()

    property alignment:
        def __get__(self):
            return GET(self.v).alignment()

    property kind:
        def __get__(self):
            return dtype_get_kind(GET(self.v))

    property string_encoding:
        def __get__(self):
            cdef string_encoding_t encoding = GET(self.v).string_encoding()
            if encoding == string_encoding_ascii:
                return "ascii"
            elif encoding == string_encoding_ucs_2:
                return "ucs_2"
            elif encoding == string_encoding_utf_8:
                return "utf_8"
            elif encoding == string_encoding_utf_16:
                return "utf_16"
            elif encoding == string_encoding_utf_32:
                return "utf_32"
            else:
                return "unknown_encoding"

    property value_dtype:
        """What this dtype looks like to calculations, printing, etc."""
        def __get__(self):
            cdef w_dtype result = w_dtype()
            SET(result.v, GET(self.v).value_dtype())
            return result

    property operand_dtype:
        """The next dtype down in the expression dtype chain."""
        def __get__(self):
            cdef w_dtype result = w_dtype()
            SET(result.v, GET(self.v).operand_dtype())
            return result

    property storage_dtype:
        """The bottom dtype in the expression chain."""
        def __get__(self):
            cdef w_dtype result = w_dtype()
            SET(result.v, GET(self.v).storage_dtype())
            return result

    property canonical_dtype:
        """The canonical version of the dtype."""
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

def make_byteswap_dtype(native_dtype, operand_dtype=None):
    """Constructs a byteswap dtype from a builtin one, with data feeding in from an optional operand dtype."""
    cdef w_dtype result = w_dtype()
    if operand_dtype is None:
        SET(result.v, dnd_make_byteswap_dtype(GET(w_dtype(native_dtype).v)))
    else:
        SET(result.v, dnd_make_byteswap_dtype(GET(w_dtype(native_dtype).v), GET(w_dtype(operand_dtype).v)))
    return result

def make_fixedbytes_dtype(int element_size, int alignment):
    """Constructs a bytes dtype with the specified element size and alignment."""
    cdef w_dtype result = w_dtype()
    SET(result.v, dnd_make_fixedbytes_dtype(element_size, alignment))
    return result

def make_convert_dtype(to_dtype, from_dtype, errmode=None):
    """Constructs a conversion dtype from the given source and destination dtypes."""
    cdef w_dtype result = w_dtype()
    SET(result.v, dnd_make_convert_dtype(GET(w_dtype(to_dtype).v), GET(w_dtype(from_dtype).v), errmode))
    return result

def make_unaligned_dtype(aligned_dtype):
    """Constructs a dtype with alignment of 1 from the given dtype."""
    cdef w_dtype result = w_dtype()
    SET(result.v, dnd_make_unaligned_dtype(GET(w_dtype(aligned_dtype).v)))
    return result

def make_fixedstring_dtype(encoding, int size):
    """Constructs a fixed-size string dtype with a specified encoding."""
    cdef w_dtype result = w_dtype()
    SET(result.v, dnd_make_fixedstring_dtype(encoding, size))
    return result

def make_string_dtype(encoding):
    """Constructs a blockref string dtype with a specified encoding."""
    cdef w_dtype result = w_dtype()
    SET(result.v, dnd_make_string_dtype(encoding))
    return result

def make_pointer_dtype(target_dtype):
    """Constructs a dtype which is a pointer to the target dtype."""
    cdef w_dtype result = w_dtype()
    SET(result.v, dnd_make_pointer_dtype(GET(w_dtype(target_dtype).v)))
    return result

def make_categorical_dtype(values):
    """Constructs a categorical dtype with the specified values as its categories."""
    cdef w_dtype result = w_dtype()
    SET(result.v, dnd_make_categorical_dtype(GET(w_ndobject(values).v)))
    return result

def factor_categorical_dtype(values):
    """Constructs a categorical dtype by factoring and sorting the unique values from the provided array."""
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
                SET(self.v, GET(self.v).cast_scalars(GET(w_dtype(dtype).v), assign_error_default))
    def __dealloc__(self):
        placement_delete(self.v)

    def debug_print(self):
        """Prints a raw representation of the ndobject data."""
        print str(ndobject_debug_print(GET(self.v)).c_str())

    def vals(self):
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

    def as_py(self):
        """Evaluates the values, and converts them into native Python types."""
        return ndobject_as_py(GET(self.v))

    def cast_scalars(self, dtype, errmode=None):
        """Converts the ndobject to the requested dtype. If dtype is an expression dtype, its expression gets applied on top of the existing data."""
        cdef w_ndobject result = w_ndobject()
        SET(result.v, ndobject_cast_scalars(GET(self.v), GET(w_dtype(dtype).v), errmode))
        return result

    def view_scalars(self, dtype):
        """Views the data of the ndobject as the requested dtype, where it makes sense."""
        cdef w_ndobject result = w_ndobject()
        SET(result.v, GET(self.v).view_scalars(GET(w_dtype(dtype).v)))
        return result

    property dtype:
        def __get__(self):
            cdef w_dtype result = w_dtype()
            SET(result.v, GET(self.v).get_dtype())
            return result

    property uniform_ndim:
        def __get__(self):
            return GET(self.v).get_dtype().get_uniform_ndim()

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

    property __array_struct__:
        # Using the __array_struct__ mechanism to expose our data to numpy
        def __get__(self):
            return ndobject_as_numpy_struct_capsule(GET(self.v))

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

def groupby(data, by, groups):
    """Produces an array containing the elements of `data`, grouped according to `by` which has corresponding shape."""
    cdef w_ndobject result = w_ndobject()
    SET(result.v, ndobject_groupby(GET(w_ndobject(data).v), GET(w_ndobject(by).v), GET(w_dtype(groups).v)))
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

cdef class w_elwise_gfunc:
    cdef elwise_gfunc_placement_wrapper v

    def __cinit__(self, bytes name):
        placement_new(self.v, name)
    def __dealloc__(self):
        placement_delete(self.v)

    property name:
        def __get__(self):
            return str(GET(self.v).get_name().c_str())

    def add_kernel(self, kernel, w_codegen_cache cgcache = default_cgcache_c):
        """Adds a kernel to the gfunc object. Currently, this means a ctypes object with prototype."""
        elwise_gfunc_add_kernel(GET(self.v), GET(cgcache.v), kernel)

    def debug_print(self):
        """Prints a raw representation of the gfunc data."""
        print str(elwise_gfunc_debug_print(GET(self.v)).c_str())

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

    def add_kernel(self, kernel, bint associative, bint commutative, identity = None, w_codegen_cache cgcache = default_cgcache_c):
        """Adds a kernel to the gfunc object. Currently, this means a ctypes object with prototype."""
        cdef w_ndobject id
        if identity is None:
            elwise_reduce_gfunc_add_kernel(GET(self.v), GET(cgcache.v), kernel, associative, commutative, ndobject())
        else:
            id = w_ndobject(identity)
            elwise_reduce_gfunc_add_kernel(GET(self.v), GET(cgcache.v), kernel, associative, commutative, GET(id.v))

    def debug_print(self):
        """Prints a raw representation of the gfunc data."""
        print str(elwise_reduce_gfunc_debug_print(GET(self.v)).c_str())

    def __call__(self, *args, **kwargs):
        """Calls the gfunc."""
        return elwise_reduce_gfunc_call(GET(self.v), args, kwargs)

cdef class w_codegen_cache:
    cdef codegen_cache_placement_wrapper v

    def __cinit__(self):
        placement_new(self.v)
    def __dealloc__(self):
        placement_delete(self.v)

    def debug_print(self):
        """Prints a raw representation of the codegen_cache data."""
        print str(codegen_cache_debug_print(GET(self.v)).c_str())

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

    def debug_print(self):
        """Prints a raw representation of the elwise_program data."""
        print str(vm_elwise_program_debug_print(GET(self.v)).c_str())

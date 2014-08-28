from __future__ import absolute_import

import sys
import ctypes
from .._pydynd import _get_lowlevel_api, _get_py_lowlevel_api
from .ctypes_types import (ArrFuncTypeDataPtr,
        CKernelBuilderStructPtr)

if sys.version_info >= (2, 7):
    c_ssize_t = ctypes.c_ssize_t
else:
    if ctypes.sizeof(ctypes.c_void_p) == 4:
        c_ssize_t = ctypes.c_int32
    else:
        c_ssize_t = ctypes.c_int64

# The low level API functions are declared with all
# void pointer parameters to avoid the copies
# of structures and values ctypes makes in various places.
class _LowLevelAPI(ctypes.Structure):
    _fields_ = [
                ('version', ctypes.c_size_t),
                # void memory_block_incref(memory_block_data *mbd);
                ('memory_block_incref',
                 ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
                # void memory_block_decref(memory_block_data *mbd);
                ('memory_block_decref',
                 ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
                # WARNING: Don't use memory_block_free unless you
                #          are correctly handling the atomic
                #          reference count decrement yourself.
                # void memory_block_free(memory_block_data *mbd);
                ('memory_block_free',
                 ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
                # void base_type_incref(const base_type *bd);
                ('base_type_incref',
                 ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
                # void base_type_decref(const base_type *bd);
                ('base_type_decref',
                 ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
                # const base_type_members *get_base_type_members(
                #                               const base_type *bd);
                ('get_base_type_members',
                 ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)),
                # void ckernel_builder_construct(void *ckb);
                ('ckernel_builder_construct',
                 ctypes.CFUNCTYPE(None, CKernelBuilderStructPtr)),
                # void ckernel_builder_destruct(void *ckb);
                ('ckernel_builder_destruct',
                 ctypes.CFUNCTYPE(None, CKernelBuilderStructPtr)),
                # void ckernel_builder_reset(void *ckb);
                ('ckernel_builder_reset',
                 ctypes.CFUNCTYPE(None, CKernelBuilderStructPtr)),
                # void ckernel_builder_ensure_capacity_leaf(void *ckb,
                #                         intptr_t requested_capacity);
                ('ckernel_builder_ensure_capacity_leaf',
                 ctypes.CFUNCTYPE(ctypes.c_int,
                        CKernelBuilderStructPtr, c_ssize_t)),
                # void ckernel_builder_ensure_capacity(void *ckb,
                #                         intptr_t requested_capacity);
                ('ckernel_builder_ensure_capacity',
                 ctypes.CFUNCTYPE(ctypes.c_int,
                        CKernelBuilderStructPtr, c_ssize_t)),
               ]

# The low level API functions are declared with all
# void pointer parameters to avoid the copies
# of structures and values ctypes makes in various places.
class _PyLowLevelAPI(ctypes.Structure):
    _fields_ = [
                ('version', ctypes.c_size_t),
                # dynd::array_preamble *get_array_ptr(WNDObject *obj);
                ('get_array_ptr',
                 ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object)),
                # const dynd::base_type *get_base_type_ptr(WDType *obj);
                ('get_base_type_ptr',
                 ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object)),
                # object array_from_ptr(dt, ptr, owner, access)
                ('array_from_ptr',
                 ctypes.PYFUNCTYPE(ctypes.py_object,
                        ctypes.py_object, ctypes.py_object,
                        ctypes.py_object, ctypes.py_object)),
                # void make_assignment_kernel(out_ckb, ckb_offset,
                #               dst_dt, dst_arrmeta,
                #               src_dt, src_arrmeta,
                #               kerntype, ectx)
                ('_make_assignment_ckernel',
                 ctypes.PYFUNCTYPE(ctypes.py_object,
                        CKernelBuilderStructPtr, c_ssize_t,
                        ctypes.py_object, ctypes.c_void_p,
                        ctypes.py_object, ctypes.c_void_p,
                        ctypes.py_object, ctypes.py_object)),
                ('make_arrfunc_from_assignment',
                 ctypes.PYFUNCTYPE(ctypes.py_object,
                        ctypes.py_object,
                        ctypes.py_object, ctypes.py_object)),
                ('make_arrfunc_from_property',
                 ctypes.PYFUNCTYPE(ctypes.py_object,
                        ctypes.py_object,
                        ctypes.py_object, ctypes.py_object)),
                # PyObject *numpy_typetuples_from_ufunc(PyObject *ufunc);
                ('numpy_typetuples_from_ufunc',
                 ctypes.PYFUNCTYPE(ctypes.py_object, ctypes.py_object)),
                # PyObject *arrfunc_from_ufunc(PyObject *ufunc,
                #   PyObject *type_tuple, int ckernel_acquires_gil);
                ('arrfunc_from_ufunc',
                 ctypes.PYFUNCTYPE(ctypes.py_object,
                        ctypes.py_object,
                        ctypes.py_object, ctypes.c_int)),
                ('lift_arrfunc',
                 ctypes.PYFUNCTYPE(ctypes.py_object,
                        ctypes.py_object)),
                ('_lift_reduction_arrfunc',
                 ctypes.PYFUNCTYPE(*([ctypes.py_object] * 10))),
                ('arrfunc_from_pyfunc',
                 ctypes.PYFUNCTYPE(ctypes.py_object,
                        ctypes.py_object, ctypes.py_object)),
                ('arrfunc_from_instantiate_pyfunc',
                 ctypes.PYFUNCTYPE(ctypes.py_object,
                        ctypes.py_object, ctypes.py_object)),
                ('make_rolling_arrfunc',
                 ctypes.PYFUNCTYPE(ctypes.py_object,
                        ctypes.py_object, ctypes.py_object)),
                ('make_builtin_mean1d_arrfunc',
                 ctypes.PYFUNCTYPE(ctypes.py_object,
                        ctypes.py_object, ctypes.py_object)),
                ('make_take_arrfunc',
                 ctypes.PYFUNCTYPE(ctypes.py_object)),
               ]

api = _LowLevelAPI.from_address(_get_lowlevel_api())
py_api = _PyLowLevelAPI.from_address(_get_py_lowlevel_api())

# The namespace consists of all the functions in the structs
__all__ = ([name for name, tp in api._fields_] +
           [name for name, tp in py_api._fields_ if not name.startswith('_')] +
           ['lift_reduction_arrfunc',
            'make_assignment_ckernel'])
for a in [api, py_api]:
    for name, tp in a._fields_:
        globals()[name] = getattr(a, name)

def lift_reduction_arrfunc(elwise_reduction, lifted_type,
                dst_initialization= None, axis=None, keepdims=False,
                associative=False, commutative=False,
                right_associative=False, reduction_identity=None):
    """
    This function creates a lifted reduction arrfunc,
    broadcasting or reducing dimensions on top of the elwise_reduction
    ckernel.

    Parameters
    ----------
    elwise_reduction : nd.array of arrfunc type
        The arrfunc object to lift. This may either be a
        unary operation which accumulates values, or a binary
        expr operation.
    lifted_type : ndt.type
        The type to lift the reduction to. This is the type of
        the src value.
    dst_initialization : nd.array of arrfunc type, optional
        If provided, this initializes an dst accumulator based
        on a src value.
    axis : int OR tuple of int, optional
        If provided, the set of axes along which to reduce.
        Defaults to a reduction along all the axes.
    keepdims: bool, optional
        If True, the reduced dimensions are kept as size-one.
        Defaults to False.
    associative: bool, optional
        If True, the elwise_reduction is based on an associative
        operation, e.g. op(op(a, b), c) == op(a, op(b, c)).
        Defaults to False.
    commutative : bool, optional
        If True, the elwise_reduction is based on a commutative
        operation, e.g. op(a, b) == op(a, b).
        Defaults to False.
    right_associative: bool, optional
        If True, the operation should associate to the right
        instead of to the left, i.e. should reduce as
        op(a, op(b, c)) instead of op(op(a, b), c). This means
        elements in the src are visited from right to left.
    reduction_identity: nd.array, optional
        If provided, this is a value ID such that in the operation
        the reduction is based on, op(a, X) == op(X, a) == a.
        A reduction of an empty list gets set to this value when
        provided.

    Returns
    -------
    nd.array of arrfunc type
        The lifted reduction arrfunc object.
    """
    return _lift_reduction_arrfunc(elwise_reduction,
                lifted_type, dst_initialization, axis, keepdims,
                associative, commutative, right_associative,
                reduction_identity)

def make_assignment_ckernel(out_ckb, ckb_offset,
                           dst_dt, dst_arrmeta,
                           src_dt, src_arrmeta,
                           kerntype, ectx=None):
    """
    This ctypes function pointer constructs a unary ckernel
    into the output ckernel_builder provided. The assignment
    constructed is from the ``src_tp`` to ``dst_tp``.

    Parameters
    ----------
    out_ckb : raw pointer
        This must point to a valid ckernel_builder object.
    ckb_offset : integer
        This is the offset within the output ckernel_builder at which
        to create the ckernel. This is nonzero when a child kernel
        is being created.
    dst_tp : ndt.type
        The destination type.
    dst_arrmeta : raw pointer or None
        A pointer to arrmeta for the destination data. This must
        remain live while the constructed ckernel exists.
    src_tp : ndt.type
        The source type
    src_arrmeta : raw pointer or None
        A pointer to arrmeta for the source data. This must
        remain live while the constructed ckernel exists.
    kerntype : 'single' or 'strided'
        Whether to create a single or strided ckernel.
    """
    return _make_assignment_ckernel(out_ckb, ckb_offset,
                                   dst_dt, dst_arrmeta,
                                   src_dt, src_arrmeta,
                                   kerntype, ectx)

# Documentation for the LowLevelAPI functions
memory_block_incref.__doc__ = """
    _lowlevel.memory_block_incref(mbd)

    This ctypes function pointer atomically increments the reference
    count of the provided dynd memory_block.

    Parameters
    ----------
    mbd : ctypes.c_void_p
        The raw address to the memory_block_data.
    """
memory_block_decref.__doc__ = """
    _lowlevel.memory_block_decref(mbd)

    This ctypes function pointer atomically decrements the reference
    count of the provided dynd memory_block, and frees it if the
    count reaches zero.

    Parameters
    ----------
    mbd : ctypes.c_void_p
        The raw address to the memory_block_data.
    """
memory_block_free.__doc__ = """
    _lowlevel.memory_block_free(mbd)

    This ctypes function pointer frees the provided dynd memory
    block, independent of its reference count. This function
    should *only* be used if the external code is handling the
    atomic decref, for example to inline the atomic increments
    and decrements in JIT compiled code.

    Parameters
    ----------
    mbd : ctypes.c_void_p
        The raw address to the memory_block_data.
    """
base_type_incref.__doc__ = """
    _lowlevel.base_type_incref(bt)

    This ctypes function pointer atomically increments the reference
    count of the provided dynd type.

    Parameters
    ----------
    bt : ctypes.c_void_p
        The raw address to the base_type.
    """
base_type_decref.__doc__ = """
    _lowlevel.base_type_decref(bt)

    This ctypes function pointer atomically decrements the reference
    count of the provided dynd type.

    Parameters
    ----------
    bt : ctypes.c_void_p
        The raw address to the base_type.
    """
get_base_type_members.__doc__ = """
    _lowlevel.get_base_type_members(bt)

    This ctypes function pointer retrieves the BaseDTypeMembers
    struct contained within the type object.

    Parameters
    ----------
    bt : ctypes.c_void_p
        The raw address to the base_type.

    Returns
    -------
    ctypes.c_void_p
        A pointer to the base dtype members struct. Use
        BaseDTypeMembers.from_address() to turn it into a struct.
    """
ckernel_builder_construct.__doc__ = """
    _lowlevel.ckernel_builder_construct(ckb)

    This ctypes function pointer initializes the provided
    block of memory as a ckernel_builder object. The memory
    pointed to by ``ckb`` must be aligned equivalent to a pointer,
    and should have size 18*sizeof(void *).

    When the object is no longer needed,
    ``_lowlevel.ckernel_builder_destruct`` must be called on the
    same pointer.

    Parameters
    ----------
    ckb : ctypes.c_void_p
        The raw address to the uninitialized ckernel_builder memory.
    """
ckernel_builder_destruct.__doc__ = """
    _lowlevel.ckernel_builder_destruct(ckb)

    This ctypes function pointer destroys the provided
    block of memory which was previously initialized using
    ``lowlevel.ckernel_builder_construct``.

    Parameters
    ----------
    ckb : ctypes.c_void_p
        The raw address to the ckernel_builder, which was previously
        initialized using ``lowlevel.ckernel_builder_construct``.
    """
ckernel_builder_reset.__doc__ = """
    _lowlevel.ckernel_builder_reset(ckb)

    This ctypes function pointer resets the provided
    block of memory which was previously initialized using
    ``lowlevel.ckernel_builder_cosntruct``, to a state equivalent
    to just being constructed.

    Parameters
    ----------
    ckb : ctypes.c_void_p
        The raw address to the ckernel_builder, which was previously
        initialized using ``lowlevel.ckernel_builder_construct``.
    """
ckernel_builder_ensure_capacity_leaf.__doc__ = """
    _lowlevel.ckernel_builder_ensure_capacity_leaf(ckb, requested_capacity)

    This ctypes function pointer ensures that the ckernel's data
    is at least the required number of bytes. It
    should only be called during the construction phase
    of the kernel when constructing a leaf ckernel.

    Parameters
    ----------
    ckb : ctypes.c_void_p
        The raw address to the ckernel_builder, which was previously
        initialized using ``lowlevel.ckernel_builder_construct``.
    requested_capacity : int
        The number of bytes that the caller requires be available in
        the ckernel.
    """
ckernel_builder_ensure_capacity.__doc__ = """
    _lowlevel.ckernel_builder_ensure_capacity(ckb, requested_capacity)

    This ctypes function pointer ensures that the ckernel's data
    is at least the required number of bytes. It
    should only be called during the construction phase
    of the kernel when constructing a ckernel with child kernels.
    This allocates enough space for the requested capacity + the
    minimal amount for a child.

    Parameters
    ----------
    ckb : ctypes.c_void_p
        The raw address to the ckernel_builder, which was previously
        initialized using ``lowlevel.ckernel_builder_construct``.
    requested_capacity : int
        The number of bytes that the caller requires be available in
        the ckernel for itself.
    """
# Documentation for the PyLowLevelAPI functions
get_array_ptr.__doc__ = """
    _lowlevel.get_array_ptr(ndarr)

    This ctypes function pointer extracts the raw dynd nd::array
    pointer out of the wrapper python nd.array object. This function
    does not validate the type of ``ndarr`, the caller must do this.

    Parameters
    ----------
    ndarr : nd.array
        The array object. This parameter is not validated.

    Returns
    -------
    ctypes.c_void_p
        The raw pointer to the nd::array object.
    """
get_base_type_ptr.__doc__ = """
    _lowlevel.get_base_type_ptr(tp)

    This ctypes function pointer extracts the raw dynd ndt::base_type
    pointer out of the wrapper python ndt.type object. This function
    does not validate the type of ``tp`, the caller must do this.

    Parameters
    ----------
    tp : ndt.type
        The type object. This parameter is not validated.

    Returns
    -------
    ctypes.c_void_p
        The raw pointer to the ndt::base_type object.
    """
array_from_ptr.__doc__ = """
    _lowlevel.array_from_ptr(tp, data_ptr, owner, access)

    This ctypes function pointer constructs an nd::array object
    from a type and a raw pointer. The ``owner`` is an object
    reference which holds on to the data pointed to by ``data_ptr``,
    and ``access` specifies permitted access.

    Parameters
    ----------
    tp : ndt.type
        The type of the array to create. This type should have
        ``arrmeta_size`` of zero.
    data_ptr : raw address
        The address of the data for the array
    owner : object
        An object reference which manages the memory pointed to
        by ``data_ptr``.
    access : 'readwrite', 'readonly', 'immutable'
        The access control of the data pointer. Note that 'immutable'
        should *only* be used when it is guaranteed that no other
        code will write to the memory while the ``owner`` reference
        is kept.

    Returns
    -------
    nd.array
        The dynd array constructed from the parameters.
    """
make_arrfunc_from_assignment.__doc__ = """
    _lowlevel.make_arrfunc_from_assignment(dst_tp, src_tp, errmode)

    This ctypes function pointer constructs a arrfunc
    object for an assignment ``src_tp`` to ``dst_tp``.

    Parameters
    ----------
    dst_tp : ndt.type
        The destination type.
    src_tp : ndt.type
        The source type

    Returns
    -------
    nd.array of arrfunc type
        The lifted arrfunc object.
    """
numpy_typetuples_from_ufunc.__doc__ = """
    _lowlevel.numpy_typetuples_from_ufunc(ufunc)

    Returns the type tuples in the functions array of the numpy ufunc.
    This works for ordinary ufuncs which are using the functions array
    in the standard way, but it is possible for a ufunc to do things
    differently, in which case this will not work correctly.

    Parameters
    ----------
    ufunc : numpy ufunc
        The ufunc to analyze.

    Returns
    -------
    list of type tuples
        A list of the type tuples for which this ufunc has functions.
    """
arrfunc_from_ufunc.__doc__ = """
    _lowlevel.arrfunc_from_ufunc(ufunc, type_tuple, ckernel_acquires_gil)

    Constructs a arrfunc object wrapping the specified
    kernel of the ufunc. The arrfunc is constructed as an 'expr'
    kernel.

    Parameters
    ----------
    ufunc : numpy ufunc
        The ufunc from which the kernel is extracted.
    type_tuple : tuple of types
        A tuple of types, providing the signature for the kernel. This
        may be one of the type tuples returned by
        ``_lowlevel.numpy_typetuples_from_ufunc``.
    ckernel_acquires_gil : bool
        If True, the resulting ckernel acquires the GIL before calling
        the ufunc's kernel. If False, it does not.

    Returns
    -------
    nd.array of arrfunc type
        The lifted arrfunc object.
    """
lift_arrfunc.__doc__ = """
    _lowlevel.lift_arrfunc(ckd)

    This function creates a lifted arrfunc, which will
    broadcast additional dimensions in arguments together.

    Parameters
    ----------
    ckd : nd.array of arrfunc type
        The arrfunc object to lift.

    Returns
    -------
    nd.array of arrfunc type
        The lifted arrfunc object.
    """
arrfunc_from_pyfunc.__doc__ = """
    _lowlevel.arrfunc_from_pyfunc(pyfunc, proto)

    Creates a dynd arrfunc from a python function that follows
    ``proto`` as its calling convention.
    """
arrfunc_from_instantiate_pyfunc.__doc__ = """
    _lowlevel.arrfunc_from_instantiate_pyfunc(instantiate_pyfunc, proto)

    Creates a dynd arrfunc from a python function that implements
    the ``instantiate`` mechanism.
    """
make_rolling_arrfunc.__doc__ = """
    _lowlevel.make_rolling_arrfunc(window_op, window_size)

    This function transforms a 1D reduction op into a rolling window op.
    """
make_builtin_mean1d_arrfunc.__doc__ = """
    _lowlevel.make_builtin_mean1d_arrfunc(tp, minp)

    This function creates a arrfunc which computes a 1D
    mean, using ``minp`` to control NaN behavior. The signature of the
    ckernel is "(strided * <tp>) -> <tp>".
    """
make_take_arrfunc.__doc__ = """
    _lowlevel.make_take_arrfunc()

    This function creates a arrfunc which applies a take
    operation along the first dimension.
    """

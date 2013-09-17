from __future__ import absolute_import

__all__ = ['CKernelPrefixStruct', 'CKernelPrefixStructPtr', 'CKernelPrefixDestructor',
        'UnarySingleOperation', 'UnaryStridedOperation',
        'ExprSingleOperation', 'ExprStridedOperation',
        'BinarySinglePredicate',
        'CKernelBuilderStruct', 'CKernelBuilderStructPtr',
        'CKernelDeferredStruct', 'CKernelDeferredStructPtr',
        'InstantiateDeferredCKernelFunction',
        'CKernel', 'CKernelBuilder', 'CKernelDeferred']

import sys
import ctypes

# ctypes.c_ssize_t/c_size_t was introduced in python 2.7
if sys.version_info >= (2, 7):
    c_ssize_t = ctypes.c_ssize_t
    c_size_t = ctypes.c_size_t
else:
    if ctypes.sizeof(ctypes.c_void_p) == 4:
        c_ssize_t = ctypes.c_int32
        c_size_t = ctypes.c_uint32
    else:
        c_ssize_t = ctypes.c_int64
        c_size_t = ctypes.c_uint64

# CKernel Prefix
CKernelPrefixDestructor = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
class CKernelPrefixStruct(ctypes.Structure):
    _fields_ = [("function", ctypes.c_void_p),
                ("destructor", CKernelPrefixDestructor)]
CKernelPrefixStructPtr = ctypes.POINTER(CKernelPrefixStruct)

class CKernel(object):
    """Wraps a ckernel prefix pointer in a callable interface.
    This object does not track ownership of where the ckernel
    came from, that must be handled by the caller.
    """
    def __init__(self, kernel_proto, ckp):
        """Constructs a CKernel from a raw ckernel prefix
        pointer, and a kernel prototype.

        Parameters
        ----------
        kernel_proto : CFUNCPTR
            The function prototype of the kernel.
        ckp : int
            A raw pointer address to a CKernelPrefixStruct.
        """
        if not issubclass(kernel_proto, ctypes._CFuncPtr):
            raise ValueError('CKernel constructor requires a ctypes '
                            'function pointer type for kernel_proto')
        self._kernel_proto = kernel_proto
        self._ckp = CKernelPrefixStruct.from_address(ckp)

    @property
    def kernel_proto(self):
        return self._kernel_proto

    @property
    def kernel_function(self):
        return ctypes.cast(self.ckp.function, self.kernel_proto)

    def __call__(self, *args):
        return self.kernel_function(*(args + (ctypes.byref(self._ckp),)))

# Unary operation function prototypes (like assignment functions)
UnarySingleOperation = ctypes.CFUNCTYPE(None,
                ctypes.c_void_p,                  # dst
                ctypes.c_void_p,                  # src
                CKernelPrefixStructPtr)           # ckp
UnaryStridedOperation = ctypes.CFUNCTYPE(None,
                ctypes.c_void_p, c_ssize_t,      # dst, dst_stride
                ctypes.c_void_p, c_ssize_t,      # src, src_stride
                c_ssize_t,                       # count
                CKernelPrefixStructPtr)          # ckp

# Expr operation function prototypes (array of src operands, # of operands is baked in)
ExprSingleOperation = ctypes.CFUNCTYPE(None,
                ctypes.c_void_p,                   # dst
                ctypes.POINTER(ctypes.c_void_p),   # src
                CKernelPrefixStructPtr)            # ckp
ExprStridedOperation = ctypes.CFUNCTYPE(None,
                ctypes.c_void_p, c_ssize_t,        # dst, dst_stride
                ctypes.POINTER(ctypes.c_void_p),
                        ctypes.POINTER(c_ssize_t), # src, src_stride
                c_ssize_t,                         # count
                CKernelPrefixStructPtr)            # ckp

# Predicates
BinarySinglePredicate = ctypes.CFUNCTYPE(ctypes.c_int, # boolean result
                ctypes.c_void_p,                   # src0
                ctypes.c_void_p,                   # src1
                CKernelPrefixStructPtr)            # ckp

# CKernel Builder
class CKernelBuilderStruct(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p),
                ("capacity", c_size_t),
                ("static_data", c_ssize_t * 16)]
CKernelBuilderStructPtr = ctypes.POINTER(CKernelBuilderStruct)

class CKernelBuilder(object):
    """Wraps a ckernel builder data structure as a python object"""
    def __init__(self):
        """Constructs an empty ckernel builder"""
        self.__ckb = CKernelBuilderStruct()
        _lowlevel.ckernel_builder_construct(self.__ckb)

    def close(self):
        if self._ckb:
            # Call the destructor
            _lowlevel.ckernel_builder_destruct(self._ckb)
            self._ckb = None

    @property
    def ckb(self):
        return self.__ckb

    def ckernel(self, kernel_proto):
        """Returns a ckernel wrapper object for the built ckernel.

        Parameters
        ----------
        kernel_proto : CFUNCPTR
            The function prototype of the kernel.
        """
        return CKernel(self._ckb.data, kernel_proto)

    def __del__(self):
        self.close()


# CKernel Deferred
InstantiateDeferredCKernelFunction = ctypes.CFUNCTYPE(None,
        ctypes.c_void_p, CKernelBuilderStructPtr, c_size_t,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32)
class CKernelDeferredStruct(ctypes.Structure):
    _fields_ = [("ckernel_funcproto", c_size_t),
                ("data_types_size", c_size_t),
                ("data_dynd_types", ctypes.c_void_p),
                ("data_ptr", ctypes.c_void_p),
                ("instantiate_func", InstantiateDeferredCKernelFunction),
                ("free_func", ctypes.CFUNCTYPE(None, ctypes.c_void_p))]
CKernelDeferredStructPtr = ctypes.POINTER(CKernelDeferredStruct)

class CKernelDeferred(object):
    def __init__(self):
        self.__ckd = CKernelDeferredStruct()

    @property
    def ckd(self):
        return self.__ckd

    def close(self):
        if self._ckd:
            # Call the free function
            if self._ckd.free_func:
                self._ckd.free_func(self._ckd.data_ptr)
            self._ckd = None

    def __del__(self):
        self.close()


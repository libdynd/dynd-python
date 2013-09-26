"""
This module provides the low level C structures of
dynd using the ctypes library.
"""

__all__ = ['BaseDTypeMembers', 'MemoryBlockData',
        'NDObjectPreamble',
        'CKernelPrefixStruct', 'CKernelPrefixStructPtr',
        'CKernelPrefixDestructor',
        'UnarySingleOperation', 'UnaryStridedOperation',
        'ExprSingleOperation', 'ExprStridedOperation',
        'BinarySinglePredicate',
        'CKernelBuilderStruct', 'CKernelBuilderStructPtr',
        'CKernelDeferredStruct', 'CKernelDeferredStructPtr',
        'InstantiateDeferredCKernelFunction',
        ]

import ctypes
import sys

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

FlagsType = ctypes.c_uint32

class BaseDTypeMembers(ctypes.Structure):
    _fields_ = [('type_id', ctypes.c_uint16),
                ('kind', ctypes.c_uint8),
                ('alignment', ctypes.c_uint8),
                ('flags', FlagsType),
                ('data_size', ctypes.c_size_t),
                ('metadata_size', ctypes.c_size_t),
                ('undim', ctypes.c_uint8)]

class MemoryBlockData(ctypes.Structure):
    # use_count is atomic<int32>, requires special treatment
    _fields_ = [('use_count', ctypes.c_int32),
                ('type', ctypes.c_uint32)]

class NDObjectPreamble(ctypes.Structure):
    _fields_ = [('memblockdata', MemoryBlockData),
                ('dtype', ctypes.c_void_p), # TODO
                ('data_pointer', ctypes.c_void_p),
                ('flags', ctypes.c_uint64),
                ('data_reference', ctypes.c_void_p)]

# CKernel Prefix
CKernelPrefixDestructor = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
class CKernelPrefixStruct(ctypes.Structure):
    _fields_ = [("function", ctypes.c_void_p),
                ("destructor", CKernelPrefixDestructor)]
CKernelPrefixStructPtr = ctypes.POINTER(CKernelPrefixStruct)

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

# CKernel Deferred
InstantiateDeferredCKernelFunction = ctypes.CFUNCTYPE(c_ssize_t,
        ctypes.c_void_p, CKernelBuilderStructPtr, c_ssize_t,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32)
class CKernelDeferredStruct(ctypes.Structure):
    _fields_ = [("ckernel_funcproto", c_size_t),
                ("data_types_size", c_size_t),
                ("data_dynd_types", ctypes.c_void_p),
                ("data_ptr", ctypes.c_void_p),
                ("instantiate_func", InstantiateDeferredCKernelFunction),
                ("free_func", ctypes.CFUNCTYPE(None, ctypes.c_void_p))]
CKernelDeferredStructPtr = ctypes.POINTER(CKernelDeferredStruct)


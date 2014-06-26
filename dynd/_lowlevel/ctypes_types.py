"""
This module provides the low level C structures of
dynd using the ctypes library.
"""

__all__ = ['BaseDTypeMembers', 'BaseDTypeMembersPtr',
        'MemoryBlockData',
        'NDArrayPreambleStruct',
        'CKernelPrefixStruct', 'CKernelPrefixStructPtr',
        'CKernelPrefixDestructor',
        'ExprSingleOperation', 'ExprStridedOperation',
        'BinarySinglePredicate',
        'CKernelBuilderStruct', 'CKernelBuilderStructPtr',
        'ArrFuncTypeData', 'ArrFuncTypeDataPtr',
        'InstantiateArrFuncFunction',
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
                ('arrmeta_size', ctypes.c_size_t),
                ('undim', ctypes.c_uint8)]
BaseDTypeMembersPtr = ctypes.POINTER(BaseDTypeMembers)

class MemoryBlockData(ctypes.Structure):
    # use_count is atomic<int32>, requires special treatment
    _fields_ = [('use_count', ctypes.c_int32),
                ('type', ctypes.c_uint32)]

class NDArrayPreambleStruct(ctypes.Structure):
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

# Expr operation function prototypes (array of src operands, # of operands is baked in)
ExprSingleOperation = ctypes.CFUNCTYPE(None,
                ctypes.c_void_p,                   # dst
                ctypes.c_void_p,                   # src
                CKernelPrefixStructPtr)            # ckp
ExprStridedOperation = ctypes.CFUNCTYPE(None,
                ctypes.c_void_p, c_ssize_t,        # dst, dst_stride
                ctypes.c_void_p,
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
                ("static_data", ctypes.c_byte * 128)]
CKernelBuilderStructPtr = ctypes.POINTER(CKernelBuilderStruct)

# ArrFunc
InstantiateArrFuncFunction = ctypes.CFUNCTYPE(c_ssize_t,
        ctypes.c_void_p, CKernelBuilderStructPtr, c_ssize_t,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_uint32, ctypes.c_void_p)
if ctypes.sizeof(ctypes.c_void_p) == 8:
    class ArrFuncTypeData(ctypes.Structure):
        _fields_ = [("func_proto", ctypes.c_void_p),
                    ("data0", ctypes.c_void_p),
                    ("data1", ctypes.c_void_p),
                    ("data2", ctypes.c_void_p),
                    ("data3", ctypes.c_void_p),
                    ("instantiate_func", InstantiateArrFuncFunction),
                    ("resolve_dst_type", ctypes.c_void_p),
                    ("resolve_dst_shape", ctypes.c_void_p),
                    ("free_func", ctypes.CFUNCTYPE(None, ctypes.c_void_p))]
else:
    class ArrFuncTypeData(ctypes.Structure):
        _fields_ = [("func_proto", ctypes.c_void_p),
                    ("data0", ctypes.c_void_p),
                    ("data1", ctypes.c_void_p),
                    ("data2", ctypes.c_void_p),
                    ("data3", ctypes.c_void_p),
                    ("data4", ctypes.c_void_p),
                    ("data5", ctypes.c_void_p),
                    ("data6", ctypes.c_void_p),
                    ("data7", ctypes.c_void_p),
                    ("data8", ctypes.c_void_p),
                    ("instantiate_func", InstantiateArrFuncFunction),
                    ("resolve_dst_type", ctypes.c_void_p),
                    ("resolve_dst_shape", ctypes.c_void_p),
                    ("free_func", ctypes.CFUNCTYPE(None, ctypes.c_void_p))]

ArrFuncTypeDataPtr = ctypes.POINTER(ArrFuncTypeData)


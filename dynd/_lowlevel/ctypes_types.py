"""
This module provides the low level C structures of
dynd using the ctypes library.
"""

__all__ = ['BaseDTypeMembers', 'MemoryBlockData',
        'NDObjectPreamble']

import ctypes

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


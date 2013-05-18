"""
This module provides the low level C structures of
dynd using the ctypes library.
"""

__all__ = ['BaseDTypeMembers', 'MemoryBlockData',
        'NDObjectPreamble', 'LowLevelAPI',
        'PyLowLevelAPI',]

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

# The low level API functions are declared with all
# void pointer parameters to avoid the copies
# of structures and values ctypes makes in various places.
class LowLevelAPI(ctypes.Structure):
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
                # void base_dtype_incref(const base_dtype *bd);
                ('base_dtype_incref',
                 ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
                # void base_dtype_decref(const base_dtype *bd);
                ('base_dtype_decref',
                 ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
                # const base_dtype_members *get_base_dtype_members(
                #                               const base_dtype *bd);
                ('get_base_dtype_members',
                 ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p)),
               ]

# The low level API functions are declared with all
# void pointer parameters to avoid the copies
# of structures and values ctypes makes in various places.
class PyLowLevelAPI(ctypes.Structure):
    _fields_ = [
                ('version', ctypes.c_size_t),
                # dynd::ndobject_preamble *get_ndobject_ptr(WNDObject *obj);
                ('get_ndobject_ptr',
                 ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object)),
                # const dynd::base_dtype *get_base_dtype_ptr(WDType *obj);
                ('get_base_dtype_ptr',
                 ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object)),
                # void make_assignment_kernel(dst_dt, src_dt, kerntype, &dki)
                ('make_assignment_kernel',
                 ctypes.PYFUNCTYPE(ctypes.py_object,
                        ctypes.py_object, ctypes.py_object,
                        ctypes.py_object, ctypes.c_void_p)),
               ]

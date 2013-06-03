from __future__ import absolute_import

__all__ = ['api', 'py_api']

import ctypes
from dynd._pydynd import _get_lowlevel_api, _get_py_lowlevel_api


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
class _PyLowLevelAPI(ctypes.Structure):
    _fields_ = [
                ('version', ctypes.c_size_t),
                # dynd::ndobject_preamble *get_ndobject_ptr(WNDObject *obj);
                ('get_ndobject_ptr',
                 ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object)),
                # const dynd::base_dtype *get_base_dtype_ptr(WDType *obj);
                ('get_base_dtype_ptr',
                 ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object)),
                # object ndobject_from_ptr(dt, ptr, owner, access)
                ('ndobject_from_ptr',
                 ctypes.PYFUNCTYPE(ctypes.py_object,
                        ctypes.py_object, ctypes.py_object,
                        ctypes.py_object, ctypes.py_object)),
                # void make_assignment_kernel(dst_dt, src_dt, kerntype, &dki)
                ('make_assignment_kernel',
                 ctypes.PYFUNCTYPE(ctypes.py_object,
                        ctypes.py_object, ctypes.py_object,
                        ctypes.py_object, ctypes.c_void_p)),
               ]

api = _LowLevelAPI.from_address(_get_lowlevel_api())
py_api = _PyLowLevelAPI.from_address(_get_py_lowlevel_api())

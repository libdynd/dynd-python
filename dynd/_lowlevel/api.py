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
                # void make_assignment_kernel(dst_dt, src_dt, kerntype, &dki)
                ('make_assignment_kernel',
                 ctypes.PYFUNCTYPE(ctypes.py_object,
                        ctypes.py_object, ctypes.py_object,
                        ctypes.py_object, ctypes.c_void_p)),
                # PyObject *numpy_typetuples_from_ufunc(PyObject *ufunc);
                ('numpy_typetuples_from_ufunc',
                 ctypes.PYFUNCTYPE(ctypes.py_object, ctypes.py_object)),
               ]

api = _LowLevelAPI.from_address(_get_lowlevel_api())
py_api = _PyLowLevelAPI.from_address(_get_py_lowlevel_api())

from __future__ import absolute_import

__all__ = [
        'CKernel', 'CKernelBuilder',
        'arrfunc_instantiate']

import ctypes
from .._pydynd import w_array as nd_array, type_of as nd_type_of, \
                      w_eval_context as nd_eval_context
from .api import api, py_api
from .ctypes_types import (CKernelPrefixStruct, NDArrayPreambleStruct,
        CKernelBuilderStruct,
        ArrFuncTypeData)
from .util import data_address_of

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
        return ctypes.cast(self._ckp.function, self.kernel_proto)

    def __call__(self, *args):
        return self.kernel_function(*(args + (ctypes.byref(self._ckp),)))

class CKernelBuilder(object):
    """Wraps a ckernel builder data structure as a python object"""
    def __init__(self, address=None):
        """
        Constructs an empty ckernel builder owned by this object,
        or builds a ckernel builder python object around a
        ckernel_builder raw pointer, borrowing its value temporarily
        """
        if address is None:
            self.__ckb = CKernelBuilderStruct()
            api.ckernel_builder_construct(self)
            self._borrowed = False
        else:
            self.__ckb = CKernelBuilderStruct.from_address(address)
            self._borrowed = True

    def close(self):
        if self.__ckb:
            if not self._borrowed:
                # Call the destructor
                api.ckernel_builder_destruct(self)
            self.__ckb = None

    def reset(self):
        # Resets the ckernel builder to its initial state
        api.ckernel_builder_reset(self)

    def ensure_capacity(self, requested_capacity):
        """Ensures that the ckernel has the requested
        capacity, together with space for a minimal child
        ckernel. Use this when building a ckernel with
        a child.

        Parameters
        ----------
        requested_capacity : int
            The number of bytes the ckernel should have.
        """
        if api.ckernel_builder_ensure_capacity(
                        self, requested_capacity) < 0:
            raise MemoryError('ckernel builder ran out of memory')

    def ensure_capacity_leaf(self, requested_capacity):
        """Ensures that the ckernel has the requested
        capacity, with no space for a child ckernel.
        Use this when creating a leaf ckernel.

        Parameters
        ----------
        requested_capacity : int
            The number of bytes the ckernel should have.
        """
        if api.ckernel_builder_ensure_capacity_leaf(
                        self, requested_capacity) < 0:
            raise MemoryError('ckernel builder ran out of memory')

    @property
    def data(self):
        """Return the pointer to the ckernel data"""
        return self.__ckb.data

    @property
    def ckb(self):
        """Returns the ckernel builder ctypes structure"""
        return self.__ckb

    @property
    def _as_parameter_(self):
        """Returns the ckernel builder byref for ctypes calls"""
        return ctypes.byref(self.__ckb)

    def ckernel(self, kernel_proto):
        """Returns a ckernel wrapper object for the built ckernel.

        Parameters
        ----------
        kernel_proto : CFUNCPTR
            The function prototype of the kernel.
        """
        return CKernel(kernel_proto, self.__ckb.data)

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

def arrfunc_instantiate(ckd, out_ckb, ckb_offset, dst_tp, dst_arrmeta,
                        src_tp, src_arrmeta, kernreq, ectx):
    if (not isinstance(ckd, nd_array) or
                nd_type_of(ckd).type_id != 'arrfunc'):
        raise TypeError('ckd must be an nd.array with type arrfunc')
    if kernreq in ["single", 0]:
        kernreq = 0
    elif kernreq in ["strided", 1]:
        kernreq = 1
    else:
        raise ValueError("invalid kernel request type %r" % kernreq)
    # Get the data pointer to the arrfunc object
    dp = NDArrayPreambleStruct.from_address(py_api.get_array_ptr(ckd)).data_pointer
    ckd_struct = ArrFuncTypeData.from_address(dp)
    if ckd_struct.instantiate_func is None:
        raise ValueError('the provided arrfunc is NULL')
    dst_tp = nd_array(dst_tp, type="type")
    src_tp = nd_array(src_tp, type="strided * type")
    src_arrmeta = nd_array(src_arrmeta, type="strided * uintptr")
    ectx_ptr = ectx._ectx_ptr if isinstance(ectx, nd_eval_context) else ectx
    ckd_struct.instantiate_func(dp,
                    out_ckb, ckb_offset,
                    data_address_of(dst_tp), dst_arrmeta,
                    data_address_of(src_tp), data_address_of(src_arrmeta),
                    kernreq, ectx_ptr)

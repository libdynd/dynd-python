from __future__ import absolute_import

__all__ = [
        'CKernel', 'CKernelBuilder',
        'arrfunc_instantiate']

import ctypes
from .api import api, py_api
from .ctypes_types import (CKernelPrefixStruct, NDArrayPreambleStruct,
        CKernelBuilderStruct,
        CKernelDeferredStruct)
from dynd import nd, ndt

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

def arrfunc_instantiate(ckd, out_ckb, ckb_offset, dynd_metadata, kerntype):
    if (not isinstance(ckd, nd.array) or
                nd.type_of(ckd).type_id != 'arrfunc'):
        raise TypeError('ckd must be an nd.array with type arrfunc')
    if kerntype in ["single", 0]:
        kerntype = 0
    elif kerntype in ["strided", 1]:
        kerntype = 1
    else:
        raise ValueError("invalid kernel request type %r" % kerntype)
    # Get the data pointer to the arrfunc object
    dp = NDArrayPreambleStruct.from_address(py_api.get_array_ptr(ckd)).data_pointer
    ckd_struct = CKernelDeferredStruct.from_address(dp)
    if ckd_struct.instantiate_func is None:
        raise ValueError('the provided arrfunc is NULL')
    ckd_struct.instantiate_func(ckd_struct.data_ptr,
                    out_ckb, ckb_offset, dynd_metadata,
                    kerntype)

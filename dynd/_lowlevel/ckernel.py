from __future__ import absolute_import

__all__ = [
        'CKernel', 'CKernelBuilder', 'CKernelDeferred']

import ctypes
from .api import api, py_api
from .ctypes_types import (CKernelPrefixStruct,
        CKernelBuilderStruct,
        CKernelDeferredStruct)

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

class CKernelBuilder(object):
    """Wraps a ckernel builder data structure as a python object"""
    def __init__(self):
        """Constructs an empty ckernel builder"""
        self.__ckb = CKernelBuilderStruct()
        api.ckernel_builder_construct(ctypes.byref(self.ckb))

    def close(self):
        if self.__ckb:
            # Call the destructor
            api.ckernel_builder_destruct(ctypes.byref(self.ckb))
            self.__ckb = None

    def ensure_capacity(requested_capacity):
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
                        ctypes.byref(self.ckb),
                        requested_capacity) < 0:
            raise MemoryError('ckernel builder ran out of memory')

    def ensure_capacity_leaf(requested_capacity):
        """Ensures that the ckernel has the requested
        capacity, with no space for a child ckernel.
        Use this when creating a leaf ckernel.

        Parameters
        ----------
        requested_capacity : int
            The number of bytes the ckernel should have.
        """
        if api.ckernel_builder_ensure_capacity_leaf(
                        ctypes.byref(self.ckb),
                        requested_capacity) < 0:
            raise MemoryError('ckernel builder ran out of memory')

    @property
    def ckb(self):
        """Returns the ckernel builder ctypes structure"""
        return self.__ckb

    def ckernel(self, kernel_proto):
        """Returns a ckernel wrapper object for the built ckernel.

        Parameters
        ----------
        kernel_proto : CFUNCPTR
            The function prototype of the kernel.
        """
        return CKernel(self.__ckb.data, kernel_proto)

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

class CKernelDeferred(object):
    def __init__(self):
        self.__ckd = CKernelDeferredStruct()

    @property
    def ckd(self):
        return self.__ckd

    def close(self):
        if self.__ckd:
            # Call the free function
            if self.__ckd.free_func:
                self.__ckd.free_func(self.__ckd.data_ptr)
            self.__ckd = None

    def __del__(self):
        self.close()


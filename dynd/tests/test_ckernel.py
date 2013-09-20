import sys
import ctypes
import unittest
from dynd import nd, ndt, _lowlevel

class TestCKernelBuilder(unittest.TestCase):
    def test_creation(self):
        with _lowlevel.ckernel.CKernelBuilder() as ckb:
            pass

    def test_allocation(self):
        with _lowlevel.ckernel.CKernelBuilder() as ckb:
            # Initially, the memory within the structure
            # is being used
            self.assertEqual(ckb.ckb.data,
                        ctypes.addressof(ckb.ckb.static_data))
            # The capacity is 16 pointer-sized objects
            self.assertEqual(ckb.ckb.capacity,
                        16 * ctypes.sizeof(ctypes.c_void_p))
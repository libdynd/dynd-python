from __future__ import print_function, absolute_import
import sys
import ctypes
import unittest
from dynd import nd, ndt, _lowlevel
import numpy as np

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
            # The capacity is 128 bytes
            initial_capacity = 128
            self.assertEqual(ckb.ckb.capacity, initial_capacity)
            # Requesting exactly the space already there should do nothing
            ckb.ensure_capacity(initial_capacity -
                            ctypes.sizeof(_lowlevel.CKernelPrefixStruct))
            self.assertEqual(ckb.ckb.data,
                        ctypes.addressof(ckb.ckb.static_data))
            self.assertEqual(ckb.ckb.capacity, initial_capacity)
            # Requesting more space should reallocate
            ckb.ensure_capacity(initial_capacity)
            self.assertTrue(ckb.ckb.data !=
                        ctypes.addressof(ckb.ckb.static_data))
            self.assertTrue(ckb.ckb.capacity >=
                    initial_capacity +
                        ctypes.sizeof(_lowlevel.CKernelPrefixStruct))

    def test_allocation_leaf(self):
        with _lowlevel.ckernel.CKernelBuilder() as ckb:
            # Initially, the memory within the structure
            # is being used
            self.assertEqual(ckb.ckb.data,
                        ctypes.addressof(ckb.ckb.static_data))
            # The capacity is 128 bytes
            initial_capacity = 128
            self.assertEqual(ckb.ckb.capacity, initial_capacity)
            # Requesting exactly the space already there should do nothing
            ckb.ensure_capacity_leaf(initial_capacity)
            self.assertEqual(ckb.ckb.data,
                        ctypes.addressof(ckb.ckb.static_data))
            self.assertEqual(ckb.ckb.capacity, initial_capacity)
            # Requesting more space should reallocate
            ckb.ensure_capacity(initial_capacity +
                            ctypes.sizeof(ctypes.c_void_p))
            self.assertTrue(ckb.ckb.data !=
                        ctypes.addressof(ckb.ckb.static_data))
            self.assertTrue(ckb.ckb.capacity >=
                    initial_capacity +
                        ctypes.sizeof(_lowlevel.CKernelPrefixStruct))

    def test_reset(self):
        with _lowlevel.ckernel.CKernelBuilder() as ckb:
            initial_capacity = 128
            # put the ckernel builder in a non-initial state
            ckb.ensure_capacity_leaf(initial_capacity + 16)
            self.assertTrue(ckb.ckb.data !=
                        ctypes.addressof(ckb.ckb.static_data))
            # verify that reset puts it back in an initial state
            ckb.reset()
            self.assertEqual(ckb.ckb.data,
                        ctypes.addressof(ckb.ckb.static_data))
            self.assertEqual(ckb.ckb.capacity, initial_capacity)

    def test_assignment_ckernel_single(self):
        with _lowlevel.ckernel.CKernelBuilder() as ckb:
            _lowlevel.make_assignment_ckernel(
                        ckb, 0,
                        ndt.float32, None, ndt.int64, None,
                        "single")
            ck = ckb.ckernel(_lowlevel.ExprSingleOperation)
            # Do an assignment using ctypes
            i64 = ctypes.c_int64(1234)
            pi64 = ctypes.pointer(i64)
            f32 = ctypes.c_float(1)
            ck(ctypes.addressof(f32), ctypes.addressof(pi64))
            self.assertEqual(f32.value, 1234.0)

    def test_assignment_ckernel_strided(self):
        with _lowlevel.ckernel.CKernelBuilder() as ckb:
            _lowlevel.make_assignment_ckernel(
                        ckb, 0,
                        ndt.float32, None, ndt.type('string[15,"A"]'), None,
                        'strided')
            ck = ckb.ckernel(_lowlevel.ExprStridedOperation)
            # Do an assignment using a numpy array
            src = np.array(['3.25', '-1000', '1e5'], dtype='S15')
            src_ptr = ctypes.c_void_p(src.ctypes.data)
            src_stride = c_ssize_t(15)
            dst = np.arange(3, dtype=np.float32)
            ck(dst.ctypes.data, 4,
               ctypes.pointer(src_ptr), ctypes.pointer(src_stride), 3)
            self.assertEqual(dst.tolist(), [3.25, -1000, 1e5])


if __name__ == '__main__':
    unittest.main()

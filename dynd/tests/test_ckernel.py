import sys
import ctypes
import unittest
from dynd import nd, ndt, _lowlevel
import numpy as np

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
            initial_capacity = 16 * ctypes.sizeof(ctypes.c_void_p)
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
            # The capacity is 16 pointer-sized objects
            initial_capacity = 16 * ctypes.sizeof(ctypes.c_void_p)
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
            initial_capacity = 16 * ctypes.sizeof(ctypes.c_void_p)
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
                        ndt.float32, None, ndt.int64, None,
                        "single", ckb)
            ck = ckb.ckernel(_lowlevel.UnarySingleOperation)
            # Do an assignment using ctypes
            i64 = ctypes.c_int64(1234)
            f32 = ctypes.c_float(1)
            ck(ctypes.addressof(f32), ctypes.addressof(i64))
            self.assertEqual(f32.value, 1234.0)

    def test_assignment_ckernel_strided(self):
        with _lowlevel.ckernel.CKernelBuilder() as ckb:
            _lowlevel.make_assignment_ckernel(
                        ndt.float32, None, ndt.type('string(15,"A")'), None,
                        'strided', ckb)
            ck = ckb.ckernel(_lowlevel.UnaryStridedOperation)
            # Do an assignment using a numpy array
            src = np.array(['3.25', '-1000', '1e5'], dtype='S15')
            dst = np.arange(3, dtype=np.float32)
            ck(dst.ctypes.data, 4, src.ctypes.data, 15, 3)
            self.assertEqual(dst.tolist(), [3.25, -1000, 1e5])

class TestCKernelDeferred(unittest.TestCase):
    def test_creation(self):
        with _lowlevel.ckernel.CKernelDeferred() as ckd:
            pass

    def test_assignment_ckernel(self):
        with _lowlevel.ckernel.CKernelDeferred() as ckd:
            _lowlevel.make_ckernel_deferred_from_assignment(
                        ndt.float32, ndt.int64,
                        "unary", "none", ckd)
            # Instantiate as a single kernel
            with _lowlevel.ckernel.CKernelBuilder() as ckb:
                meta = (ctypes.c_void_p * 2)()
                ckd.instantiate(ckb, 0, meta, "single")
                ck = ckb.ckernel(_lowlevel.UnarySingleOperation)
                # Do an assignment using ctypes
                i64 = ctypes.c_int64(1234)
                f32 = ctypes.c_float(1)
                ck(ctypes.addressof(f32), ctypes.addressof(i64))
                self.assertEqual(f32.value, 1234.0)
            # Instantiate as a strided kernel
            with _lowlevel.ckernel.CKernelBuilder() as ckb:
                meta = (ctypes.c_void_p * 2)()
                ckd.instantiate(ckb, 0, meta, "strided")
                ck = ckb.ckernel(_lowlevel.UnaryStridedOperation)
                # Do an assignment using ctypes
                i64 = (ctypes.c_int64 * 3)()
                for i, v in enumerate([3,7,21]):
                    i64[i] = v
                f32 = (ctypes.c_float * 3)()
                ck(ctypes.addressof(f32), 4,
                            ctypes.addressof(i64), 8,
                            3)
                self.assertEqual([f32[i] for i in range(3)], [3,7,21])



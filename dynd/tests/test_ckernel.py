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
                        ckb, 0,
                        ndt.float32, None, ndt.int64, None,
                        "unary", "single")
            ck = ckb.ckernel(_lowlevel.UnarySingleOperation)
            # Do an assignment using ctypes
            i64 = ctypes.c_int64(1234)
            f32 = ctypes.c_float(1)
            ck(ctypes.addressof(f32), ctypes.addressof(i64))
            self.assertEqual(f32.value, 1234.0)

    def test_assignment_ckernel_strided(self):
        with _lowlevel.ckernel.CKernelBuilder() as ckb:
            _lowlevel.make_assignment_ckernel(
                        ckb, 0,
                        ndt.float32, None, ndt.type('string[15,"A"]'), None,
                        'unary', 'strided')
            ck = ckb.ckernel(_lowlevel.UnaryStridedOperation)
            # Do an assignment using a numpy array
            src = np.array(['3.25', '-1000', '1e5'], dtype='S15')
            dst = np.arange(3, dtype=np.float32)
            ck(dst.ctypes.data, 4, src.ctypes.data, 15, 3)
            self.assertEqual(dst.tolist(), [3.25, -1000, 1e5])

class TestCKernelDeferred(unittest.TestCase):
    def test_creation(self):
        ckd = nd.empty('ckernel_deferred')
        self.assertEqual(nd.type_of(ckd).type_id, 'ckernel_deferred')
        # Test there is a string version of a NULL ckernel_deferred
        self.assertTrue(str(ckd) != '')
        self.assertEqual(nd.as_py(ckd.types), [])
        # Test there is a string version of an initialized ckernel_deferred
        ckd = _lowlevel.make_ckernel_deferred_from_assignment(
                    ndt.float32, ndt.int64,
                    "unary", "none")
        self.assertTrue(str(ckd) != '')
        self.assertEqual(nd.as_py(ckd.types), [ndt.float32, ndt.int64])

    def test_assignment_ckernel(self):
        ckd = _lowlevel.make_ckernel_deferred_from_assignment(
                    ndt.float32, ndt.int64,
                    "unary", "none")
        self.assertEqual(nd.as_py(ckd.types), [ndt.float32, ndt.int64])
        # Instantiate as a single kernel
        with _lowlevel.ckernel.CKernelBuilder() as ckb:
            meta = (ctypes.c_void_p * 2)()
            _lowlevel.ckernel_deferred_instantiate(ckd, ckb, 0, meta, "single")
            ck = ckb.ckernel(_lowlevel.UnarySingleOperation)
            # Do an assignment using ctypes
            i64 = ctypes.c_int64(1234)
            f32 = ctypes.c_float(1)
            ck(ctypes.addressof(f32), ctypes.addressof(i64))
            self.assertEqual(f32.value, 1234.0)
        # Instantiate as a strided kernel
        with _lowlevel.ckernel.CKernelBuilder() as ckb:
            meta = (ctypes.c_void_p * 2)()
            _lowlevel.ckernel_deferred_instantiate(ckd, ckb, 0, meta, "strided")
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

    def check_from_numpy_int32_add(self, requiregil):
        # Get int32 add as a ckernel_deferred
        ckd = _lowlevel.ckernel_deferred_from_ufunc(np.add,
                        (np.int32, np.int32, np.int32),
                        requiregil)
        self.assertEqual(nd.as_py(ckd.types), [ndt.int32]*3)
        # Instantiate as a single kernel
        with _lowlevel.ckernel.CKernelBuilder() as ckb:
            meta = (ctypes.c_void_p * 3)()
            _lowlevel.ckernel_deferred_instantiate(ckd, ckb, 0, meta, "single")
            ck = ckb.ckernel(_lowlevel.ExprSingleOperation)
            a = ctypes.c_int32(10)
            b = ctypes.c_int32(21)
            c = ctypes.c_int32(0)
            src = (ctypes.c_void_p * 2)()
            src[0] = ctypes.addressof(a)
            src[1] = ctypes.addressof(b)
            ck(ctypes.addressof(c), src)
            self.assertEqual(c.value, 31)
        # Instantiate as a strided kernel
        with _lowlevel.ckernel.CKernelBuilder() as ckb:
            meta = (ctypes.c_void_p * 3)()
            _lowlevel.ckernel_deferred_instantiate(ckd, ckb, 0, meta, "strided")
            ck = ckb.ckernel(_lowlevel.ExprStridedOperation)
            a = (ctypes.c_int32 * 3)()
            b = (ctypes.c_int32 * 3)()
            c = (ctypes.c_int32 * 3)()
            for i, v in enumerate([1,4,6]):
                a[i] = v
            for i, v in enumerate([3, -1, 12]):
                b[i] = v
            src = (ctypes.c_void_p * 2)()
            src[0] = ctypes.addressof(a)
            src[1] = ctypes.addressof(b)
            strides = (c_ssize_t * 2)()
            strides[0] = strides[1] = 4
            ck(ctypes.addressof(c), 4, src, strides, 3)
            self.assertEqual(c[0], 4)
            self.assertEqual(c[1], 3)
            self.assertEqual(c[2], 18)

    def test_from_numpy_int32_add_nogil(self):
        self.check_from_numpy_int32_add(False)

    def test_from_numpy_int32_add_withgil(self):
        self.check_from_numpy_int32_add(True)

    def test_lift_ckernel(self):
        # First get a ckernel from numpy
        requiregil = False
        ckd = _lowlevel.ckernel_deferred_from_ufunc(np.ldexp,
                        (np.float64, np.float64, np.int32),
                        requiregil)
        self.assertEqual(nd.as_py(ckd.types),
                        [ndt.float64, ndt.float64, ndt.int32])

        # Now lift it
        ckd_lifted = _lowlevel.lift_ckernel_deferred(ckd,
                        ['var * var * float64', 'strided * var * float64',
                         'strided * 1 * int32'])
        self.assertEqual(nd.as_py(ckd_lifted.types),
                        [ndt.type(x) for x in ['var * var * float64',
                                    'strided * var * float64', 'strided * 1 * int32']])
        # Create some compatible arguments
        out = nd.empty('var * var * float64')
        in0 = nd.array([[1, 2, 3], [4, 5], [6], [7,9,10]], type='strided * var * float64')
        in1 = nd.array([[-1], [10], [100], [-12]], type='strided * 1 * int32')
        # Instantiate and call the kernel on these arguments
        ckd_lifted.__call__(out, in0, in1)
        # Verify that we got the expected result
        self.assertEqual(nd.as_py(out),
                    [[0.5, 1.0, 1.5],
                     [4096.0, 5120.0],
                     [float(6*2**100)],
                     [0.001708984375, 0.002197265625, 0.00244140625]])

    def test_ckernel_deferred_from_pyfunc(self):
        # Test wrapping make_assignment_ckernel as a deferred ckernel
        def instantiate_assignment(out_ckb, ckb_offset, types, meta,
                                   kerntype, ectx):
            out_ckb = _lowlevel.CKernelBuilderStruct.from_address(out_ckb)
            return _lowlevel.make_assignment_ckernel(out_ckb, ckb_offset,
                            types[0], meta[0],
                            types[1], meta[1],
                            'expr', kerntype, ectx)
        ckd = _lowlevel.ckernel_deferred_from_pyfunc(instantiate_assignment,
                        [ndt.string, ndt.date])
        self.assertEqual(nd.as_py(ckd.types), [ndt.string, ndt.date])
        out = nd.empty(ndt.string)
        in0 = nd.array('2012-11-05', ndt.date)
        ckd.__call__(out, in0)
        self.assertEqual(nd.as_py(out), '2012-11-05')
        # Also test it as a lifted kernel
        ckd_lifted = _lowlevel.lift_ckernel_deferred(ckd,
                        ['3 * var * string', '3 * var * date'])
        self.assertEqual(nd.as_py(ckd_lifted.types),
                    [ndt.type('3 * var * string'), ndt.type('3 * var * date')])
        out = nd.empty('3 * var * string')
        from datetime import date
        in0 = nd.array([['2013-03-11', date(2010, 10, 10)],
                        [date(1999, 12, 31)],
                        []], type='3 * var * date')
        ckd_lifted.__call__(out, in0)
        self.assertEqual(nd.as_py(out),
                        [['2013-03-11', '2010-10-10'],
                         ['1999-12-31'], []])


class TestLiftReductionCKernelDeferred(unittest.TestCase):
    def test_sum_1d(self):
        # Use the numpy add ufunc for this lifting test
        ckd = _lowlevel.ckernel_deferred_from_ufunc(np.add,
                        (np.int32, np.int32, np.int32),
                        False)
        in0 = nd.array([3, 12, -5, 10, 2])
        # Simple lift
        sum = _lowlevel.lift_reduction_ckernel_deferred(ckd, 'strided * int32')
        out = nd.empty(ndt.int32)
        sum.__call__(out, in0)
        self.assertEqual(nd.as_py(out), 22)
        # Lift with keepdims
        sum = _lowlevel.lift_reduction_ckernel_deferred(ckd, 'strided * int32',
                                                        keepdims=True)
        out = nd.empty(1, ndt.int32)
        sum.__call__(out, in0)
        self.assertEqual(nd.as_py(out), [22])

    def test_sum_2d_axisall(self):
        # Use the numpy add ufunc for this lifting test
        ckd = _lowlevel.ckernel_deferred_from_ufunc(np.add,
                        (np.int32, np.int32, np.int32),
                        False)
        in0 = nd.array([[3, 12, -5], [10, 2, 3]])
        # Simple lift
        sum = _lowlevel.lift_reduction_ckernel_deferred(ckd,
                                                 'strided * strided * int32',
                                                 commutative=True,
                                                 associative=True)
        out = nd.empty(ndt.int32)
        sum.__call__(out, in0)
        self.assertEqual(nd.as_py(out), 25)

    def test_sum_2d_axis0(self):
        # Use the numpy add ufunc for this lifting test
        ckd = _lowlevel.ckernel_deferred_from_ufunc(np.add,
                        (np.int32, np.int32, np.int32),
                        False)
        in0 = nd.array([[3, 12, -5], [10, 2, 3]])
        # Reduce along axis 0
        sum = _lowlevel.lift_reduction_ckernel_deferred(ckd,
                                                 'strided * strided * int32',
                                                 axis=0,
                                                 commutative=True,
                                                 associative=True)
        out = nd.empty(3, ndt.int32)
        sum.__call__(out, in0)
        self.assertEqual(nd.as_py(out), [13, 14, -2])

    def test_sum_2d_axis1(self):
        # Use the numpy add ufunc for this lifting test
        ckd = _lowlevel.ckernel_deferred_from_ufunc(np.add,
                        (np.int32, np.int32, np.int32),
                        False)
        in0 = nd.array([[3, 12, -5], [10, 2, 3]])
        # Reduce along axis 1
        sum = _lowlevel.lift_reduction_ckernel_deferred(ckd,
                                                 'strided * strided * int32',
                                                 axis=1,
                                                 commutative=True,
                                                 associative=True)
        out = nd.empty(2, ndt.int32)
        sum.__call__(out, in0)
        self.assertEqual(nd.as_py(out), [10, 15])

if __name__ == '__main__':
    unittest.main()

from __future__ import print_function, absolute_import
import sys
import ctypes
import unittest
from dynd import nd, ndt, _lowlevel
import numpy as np

if sys.version_info >= (2, 7):
    c_ssize_t = ctypes.c_ssize_t
else:
    if ctypes.sizeof(ctypes.c_void_p) == 4:
        c_ssize_t = ctypes.c_int32
    else:
        c_ssize_t = ctypes.c_int64

class TestArrFunc(unittest.TestCase):
    def test_creation(self):
        af = nd.empty('arrfunc')
        self.assertEqual(nd.type_of(af).type_id, 'arrfunc')
        # Test there is a string version of a NULL arrfunc
        self.assertTrue(str(af) != '')
        self.assertEqual(nd.as_py(af.proto), ndt.type())
        # Test there is a string version of an initialized arrfunc
        af = _lowlevel.make_arrfunc_from_assignment(
                    ndt.float32, ndt.int64, "nocheck")
        self.assertTrue(str(af) != '')
        self.assertEqual(nd.as_py(af.proto), ndt.type("(int64) -> float32"))

    def test_arrfunc_constructor(self):
        af = nd.arrfunc(lambda x, y : [x, y], '(int, int) -> {x:int, y:int}')
        a = af(1, 10)
        self.assertEqual(nd.as_py(a), {'x': 1, 'y': 10})

    def test_assignment_arrfunc(self):
        af = _lowlevel.make_arrfunc_from_assignment(
                    ndt.float32, ndt.int64, "nocheck")
        self.assertEqual(nd.as_py(af.proto), ndt.type("(int64) -> float32"))
        a = nd.array(1234, type=ndt.int64)
        b = af(a)
        self.assertEqual(nd.type_of(b), ndt.float32)
        self.assertEqual(nd.as_py(b), 1234)
        # Instantiate as a strided kernel
        with _lowlevel.ckernel.CKernelBuilder() as ckb:
            meta = (ctypes.c_void_p * 2)()
            ectx = nd.eval_context()
            _lowlevel.arrfunc_instantiate(af, ckb, 0, ndt.float32, 0,
                                          [ndt.int64], [0], "strided",
                                          ectx._ectx_ptr)
            ck = ckb.ckernel(_lowlevel.ExprStridedOperation)
            # Do an assignment using ctypes
            i64 = (ctypes.c_int64 * 3)()
            for i, v in enumerate([3,7,21]):
                i64[i] = v
            pi64 = ctypes.pointer(i64)
            i64_stride = c_ssize_t(8)
            f32 = (ctypes.c_float * 3)()
            ck(ctypes.addressof(f32), 4,
                        ctypes.addressof(pi64), ctypes.pointer(i64_stride),
                        3)
            self.assertEqual([f32[i] for i in range(3)], [3,7,21])

    def check_from_numpy_int32_add(self, requiregil):
        # Get int32 add as an arrfunc
        af = _lowlevel.arrfunc_from_ufunc(np.add,
                        (np.int32, np.int32, np.int32),
                        requiregil)
        self.assertEqual(nd.as_py(af.proto),
                         ndt.type("(int32, int32) -> int32"))

        a = nd.array(10, ndt.int32)
        b = nd.array(21, ndt.int32)
        c = af(a, b)
        self.assertEqual(nd.type_of(c), ndt.int32)
        self.assertEqual(nd.as_py(c), 31)

        af_lift = _lowlevel.lift_arrfunc(af)
        a = af_lift([[1], [2, 3], [4, 5, 6]], [[5, 10], [2], [1, 5, 1]])
        self.assertEqual(nd.type_of(a), ndt.type('strided * var * int'))
        self.assertEqual(nd.as_py(a), [[6, 11], [4, 5], [5, 10, 7]])

    def test_from_numpy_int32_add_nogil(self):
        self.check_from_numpy_int32_add(False)

    def test_from_numpy_int32_add_withgil(self):
        self.check_from_numpy_int32_add(True)

    def test_lift_arrfunc(self):
        # First get a ckernel from numpy
        requiregil = False
        af = _lowlevel.arrfunc_from_ufunc(np.ldexp,
                        (np.float64, np.float64, np.int32),
                        requiregil)
        self.assertEqual(nd.as_py(af.proto),
                         ndt.type("(float64, int32) -> float64"))

        # Now lift it
        af_lifted = _lowlevel.lift_arrfunc(af)
        self.assertEqual(nd.as_py(af_lifted.proto),
                         ndt.type("(Dims... * float64, Dims... * int32) -> Dims... * float64"))
        # Create some compatible arguments
        in0 = nd.array([[1, 2, 3], [4, 5], [6], [7,9,10]],
                       type='strided * var * float64')
        in1 = nd.array([[-1], [10], [100], [-12]], type='strided * 1 * int32')
        # Instantiate and call the kernel on these arguments
        out = af_lifted(in0, in1)
        # Verify that we got the expected result
        self.assertEqual(nd.as_py(out),
                    [[0.5, 1.0, 1.5],
                     [4096.0, 5120.0],
                     [float(6*2**100)],
                     [0.001708984375, 0.002197265625, 0.00244140625]])

    def test_arrfunc_from_pyfunc(self):
        # Create an arrfunc out of a python function
        def myweightedsum(wt, a):
            wt = nd.as_py(wt)
            a = nd.as_py(a)
            return sum(x * y for x, y in zip(wt, a)) / sum(wt)
        af = _lowlevel.arrfunc_from_pyfunc(myweightedsum,
                                           "(var * real, var * real) -> real")
        in0 = nd.array([0.5, 1.0, 0.5], type="var * real")
        in1 = nd.array([1, 3, 5], type="var * real")
        out = af(in0, in1)
        self.assertEqual(nd.as_py(out), (0.5 + 3.0 + 2.5) / 2.0)
        # Also test it as a lifted kernel
        af_lifted = _lowlevel.lift_arrfunc(af)
        in0 = nd.array([[0.25, 0.75], [0.5, 1.0, 0.5], [1.0]],
                       type="strided * var * real")
        in1 = nd.array([[1, 3], [1, 3, 5], [5]],
                       type="strided * var * real")
        out = af_lifted(in0, in1)
        self.assertEqual(nd.as_py(out),
                         [(0.25 + 0.75 * 3),
                          (0.5 + 3.0 + 2.5) / 2.0,
                          5.0])

    def test_arrfunc_from_instantiate_pyfunc(self):
        # Test wrapping make_assignment_ckernel as an arrfunc
        def instantiate_assignment(out_ckb, ckb_offset, dst_tp, dst_arrmeta,
                                   src_tp, src_arrmeta, kernreq, ectx):
            out_ckb = _lowlevel.CKernelBuilderStruct.from_address(out_ckb)
            return _lowlevel.make_assignment_ckernel(out_ckb, ckb_offset,
                            dst_tp, dst_arrmeta,
                            src_tp[0], src_arrmeta[0],
                            kernreq, ectx)
        af = _lowlevel.arrfunc_from_instantiate_pyfunc(
                    instantiate_assignment, "(date) -> string")
        self.assertEqual(nd.as_py(af.proto), ndt.type("(date) -> string"))
        in0 = nd.array('2012-11-05', ndt.date)
        out = af(in0)
        self.assertEqual(nd.as_py(out), '2012-11-05')
        # Also test it as a lifted kernel
        af_lifted = _lowlevel.lift_arrfunc(af)
        self.assertEqual(nd.as_py(af_lifted.proto),
                         ndt.type("(Dims... * date) -> Dims... * string"))
        from datetime import date
        in0 = nd.array([['2013-03-11', date(2010, 10, 10)],
                        [date(1999, 12, 31)],
                        []], type='3 * var * date')
        out = af_lifted(in0)
        self.assertEqual(nd.as_py(out),
                        [['2013-03-11', '2010-10-10'],
                         ['1999-12-31'], []])

class TestLiftReductionArrFunc(unittest.TestCase):
    def test_sum_1d(self):
        # Use the numpy add ufunc for this lifting test
        af = _lowlevel.arrfunc_from_ufunc(np.add,
                        (np.int32, np.int32, np.int32),
                        False)
        in0 = nd.array([3, 12, -5, 10, 2])
        # Simple lift
        sum = _lowlevel.lift_reduction_arrfunc(af, 'strided * int32')
        out = nd.empty(ndt.int32)
        sum.execute(out, in0)
        self.assertEqual(nd.as_py(out), 22)
        # Lift with keepdims
        sum = _lowlevel.lift_reduction_arrfunc(af, 'strided * int32',
                                                        keepdims=True)
        out = nd.empty(1, ndt.int32)
        sum.execute(out, in0)
        self.assertEqual(nd.as_py(out), [22])

    def test_sum_2d_axisall(self):
        # Use the numpy add ufunc for this lifting test
        af = _lowlevel.arrfunc_from_ufunc(np.add,
                        (np.int32, np.int32, np.int32),
                        False)
        in0 = nd.array([[3, 12, -5], [10, 2, 3]])
        # Simple lift
        sum = _lowlevel.lift_reduction_arrfunc(af,
                                                 'strided * strided * int32',
                                                 commutative=True,
                                                 associative=True)
        out = nd.empty(ndt.int32)
        sum.execute(out, in0)
        self.assertEqual(nd.as_py(out), 25)

    def test_sum_2d_axis0(self):
        # Use the numpy add ufunc for this lifting test
        af = _lowlevel.arrfunc_from_ufunc(np.add,
                        (np.int32, np.int32, np.int32),
                        False)
        in0 = nd.array([[3, 12, -5], [10, 2, 3]])
        # Reduce along axis 0
        sum = _lowlevel.lift_reduction_arrfunc(af,
                                                 'strided * strided * int32',
                                                 axis=0,
                                                 commutative=True,
                                                 associative=True)
        out = nd.empty(3, ndt.int32)
        sum.execute(out, in0)
        self.assertEqual(nd.as_py(out), [13, 14, -2])

    def test_sum_2d_axis1(self):
        # Use the numpy add ufunc for this lifting test
        af = _lowlevel.arrfunc_from_ufunc(np.add,
                        (np.int32, np.int32, np.int32),
                        False)
        # Reduce along axis 1
        sum = _lowlevel.lift_reduction_arrfunc(af,
                                                 'strided * strided * int32',
                                                 axis=1,
                                                 commutative=True,
                                                 associative=True)
        in0 = nd.array([[3, 12, -5], [10, 2, 3]])
        out = nd.empty(2, ndt.int32)
        sum.execute(out, in0)
        self.assertEqual(nd.as_py(out), [10, 15])


class TestRollingArrFunc(unittest.TestCase):
    def test_diff_op(self):
        # Use the numpy subtract ufunc for this lifting test
        af = _lowlevel.arrfunc_from_ufunc(np.subtract,
                        (np.float64, np.float64, np.float64),
                        False)
        # Lift it to 1D
        diff_1d = _lowlevel.lift_reduction_arrfunc(af,
                                                 'strided * float64',
                                                 axis=0,
                                                 commutative=False,
                                                 associative=False)
        # Apply it as a rolling op
        diff = _lowlevel.make_rolling_arrfunc(diff_1d, 2)
        in0 = nd.array([1.5, 3.25, 7, -3.5, 1.25])
        out = diff(in0)
        result = nd.as_py(out)
        self.assertTrue(np.isnan(result[0]))
        self.assertEqual(result[1:],
                         [3.25 - 1.5 , 7 - 3.25, -3.5 - 7, 1.25 - -3.5])

    def test_rolling_mean(self):
        mean_1d = _lowlevel.make_builtin_mean1d_arrfunc('float64', -1)
        rolling_mean = _lowlevel.make_rolling_arrfunc(mean_1d, 4)
        in0 = nd.array([3.0, 2, 1, 3, 8, nd.nan, nd.nan])
        out = rolling_mean(in0)
        result = nd.as_py(out)
        self.assertTrue(np.all(np.isnan(result[:3])))
        self.assertTrue(np.isnan(result[-1]))
        self.assertEqual(result[3:-1], [9.0/4, 14.0/4, 12.0/3])


if __name__ == '__main__':
    unittest.main()

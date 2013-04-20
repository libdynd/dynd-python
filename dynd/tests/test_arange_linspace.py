import sys
import unittest
from dynd import nd, ndt

class TestArange(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(nd.as_py(nd.arange(10)), list(range(10)))
        self.assertEqual(nd.as_py(nd.arange(5, 10)), list(range(5, 10)))
        self.assertEqual(nd.as_py(nd.arange(5, 10, 3)), list(range(5, 10, 3)))
        self.assertEqual(nd.as_py(nd.arange(10, 5, -1)), list(range(10, 5, -1)))
        self.assertEqual(nd.as_py(nd.arange(stop=10, step=2)), list(range(0, 10, 2)))

    def test_default_dtype(self):
        # Defaults to int32 when given ints
        self.assertEqual(nd.arange(10).udtype, ndt.int32)
        # Except if the input numbers don't fit, then returns int64
        self.assertEqual(nd.arange(2**32, 2**32+10).udtype, ndt.int64)
        self.assertEqual(nd.arange(-2**32, -2**32+10).udtype, ndt.int64)
        # Gives float64 when given floats
        self.assertEqual(nd.arange(10.0).udtype, ndt.float64)

    def test_specified_dtype(self):
        # Must return the requested type
        self.assertRaises(RuntimeError, nd.arange, 10, dtype=ndt.bool)
        self.assertEqual(nd.arange(10, dtype=ndt.int8).udtype, ndt.int8)
        self.assertEqual(nd.arange(10, dtype=ndt.int16).udtype, ndt.int16)
        self.assertEqual(nd.arange(10, dtype=ndt.int32).udtype, ndt.int32)
        self.assertEqual(nd.arange(10, dtype=ndt.int64).udtype, ndt.int64)
        self.assertEqual(nd.arange(10, dtype=ndt.uint8).udtype, ndt.uint8)
        self.assertEqual(nd.arange(10, dtype=ndt.uint16).udtype, ndt.uint16)
        self.assertEqual(nd.arange(10, dtype=ndt.uint32).udtype, ndt.uint32)
        self.assertEqual(nd.arange(10, dtype=ndt.uint64).udtype, ndt.uint64)
        self.assertEqual(nd.arange(10, dtype=ndt.float32).udtype, ndt.float32)
        self.assertEqual(nd.arange(10, dtype=ndt.float64).udtype, ndt.float64)
        # Maybe in the future add complex support when start.imag == stop.imag
        # and step.imag == 0?
        self.assertRaises(RuntimeError, nd.arange, 10, dtype=ndt.cfloat32)
        self.assertRaises(RuntimeError, nd.arange, 10, dtype=ndt.cfloat64)
        # Float/complex should convert when the dtype is specified
        self.assertEqual(nd.arange(10.0, dtype=ndt.uint16).udtype, ndt.uint16)
        self.assertEqual(nd.arange(1.0, step=0.5+0j, dtype=ndt.float32).udtype, ndt.float32)

    def test_float_step(self):
        # Should produce the correct count for 1.0/int steps
        for i in range(1, 32):
            a = nd.arange(1.0, step=1.0/i)
            self.assertEqual(len(a), i)
            self.assertEqual(nd.as_py(a[0]), 0)
        # For powers of two, should be getting exact answers
        for i in range(5):
            a = nd.arange(1.0, step=1.0/2**i)
            self.assertEqual(nd.as_py(a), [float(x)/2**i for x in range(2**i)])

    def test_cast_errors(self):
        # If a dtype is specified, the inputs must be convertible
        self.assertRaises(RuntimeError, nd.arange, 1.5, dtype=ndt.int32)
        self.assertRaises(RuntimeError, nd.arange, 1j, 10, 1, dtype=ndt.int32)
        self.assertRaises(RuntimeError, nd.arange, 0, 1j, 1, dtype=ndt.int32)
        self.assertRaises(RuntimeError, nd.arange, 0, 10, 1j, dtype=ndt.int32)

class TestLinspace(unittest.TestCase):
    def test_simple(self):
        # Default is a count of 50. For these simple cases of integers,
        # the result should be exact
        self.assertEqual(nd.as_py(nd.linspace(0, 49)), range(50))
        self.assertEqual(nd.as_py(nd.linspace(49, 0)), range(49, -1, -1))
        self.assertEqual(nd.as_py(nd.linspace(0, 10, count=11)), range(11))
        self.assertEqual(nd.as_py(nd.linspace(1, -1, count=2)), [1, -1])
        self.assertEqual(nd.as_py(nd.linspace(1j, 50j)), [i*1j for i in range(1, 51)])

    def test_default_dtype(self):
        # Defaults to float64 when given ints
        self.assertEqual(nd.linspace(0, 1).udtype, ndt.float64)
        # Gives float64 when given floats
        self.assertEqual(nd.linspace(0, 1.0).udtype, ndt.float64)
        self.assertEqual(nd.linspace(0.0, 1).udtype, ndt.float64)
        # Gives cfloat64 when given complex
        self.assertEqual(nd.linspace(1.0, 1.0j).udtype, ndt.cfloat64)
        self.assertEqual(nd.linspace(0.0j, 1.0).udtype, ndt.cfloat64)

    def test_specified_dtype(self):
        # Linspace only supports real-valued outputs
        self.assertRaises(RuntimeError, nd.linspace, 0, 1, dtype=ndt.bool)
        self.assertRaises(RuntimeError, nd.linspace, 0, 1, dtype=ndt.int8)
        self.assertRaises(RuntimeError, nd.linspace, 0, 1, dtype=ndt.int16)
        self.assertRaises(RuntimeError, nd.linspace, 0, 1, dtype=ndt.int32)
        self.assertRaises(RuntimeError, nd.linspace, 0, 1, dtype=ndt.int64)
        self.assertRaises(RuntimeError, nd.linspace, 0, 1, dtype=ndt.uint8)
        self.assertRaises(RuntimeError, nd.linspace, 0, 1, dtype=ndt.uint16)
        self.assertRaises(RuntimeError, nd.linspace, 0, 1, dtype=ndt.uint32)
        self.assertRaises(RuntimeError, nd.linspace, 0, 1, dtype=ndt.uint64)
        # Should obey the float/complex type requests
        self.assertEqual(nd.linspace(0, 1, dtype=ndt.float32).udtype, ndt.float32)
        self.assertEqual(nd.linspace(0, 1, dtype=ndt.float64).udtype, ndt.float64)
        self.assertEqual(nd.linspace(0, 1, dtype=ndt.cfloat32).udtype, ndt.cfloat32)
        self.assertEqual(nd.linspace(0, 1, dtype=ndt.cfloat64).udtype, ndt.cfloat64)

    def test_cast_errors(self):
        # If a dtype is specified, the inputs must be convertible
        self.assertRaises(RuntimeError, nd.linspace, 0j, 1j, dtype=ndt.float32)
        self.assertRaises(RuntimeError, nd.linspace, 0j, 1j, dtype=ndt.float64)

if __name__ == '__main__':
    unittest.main()

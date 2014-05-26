import sys
import unittest
from dynd import nd, ndt

class TestInt128(unittest.TestCase):
    def test_pyconvert(self):
        # Conversions to/from python longs
        a = nd.empty(ndt.int128)
        a[...] = 1
        self.assertEqual(nd.as_py(a), 1)
        a[...] = 12345
        self.assertEqual(nd.as_py(a), 12345)
        a[...] = -12345
        self.assertEqual(nd.as_py(a), -12345)
        a[...] = -2**127
        self.assertEqual(nd.as_py(a), -2**127)
        a[...] = 2**127 - 1
        self.assertEqual(nd.as_py(a), 2**127 - 1)

    def test_pyconvert_overflow(self):
        a = nd.empty(ndt.int128)
        def assign_val(x, val):
            x[...] = val
        self.assertRaises(OverflowError, assign_val, a, -2**127 - 1)
        self.assertRaises(OverflowError, assign_val, a, 2**127)

class TestUInt128(unittest.TestCase):
    def test_pyconvert(self):
        # Conversions to/from python longs
        a = nd.empty(ndt.uint128)
        a[...] = 1
        self.assertEqual(nd.as_py(a), 1)
        a[...] = 12345
        self.assertEqual(nd.as_py(a), 12345)
        a[...] = 0
        self.assertEqual(nd.as_py(a), 0)
        a[...] = 2**128 - 1
        self.assertEqual(nd.as_py(a), 2**128 - 1)

    def test_pyconvert_overflow(self):
        a = nd.empty(ndt.uint128)
        def assign_val(x, val):
            x[...] = val
        self.assertRaises(OverflowError, assign_val, a, -1)
        self.assertRaises(OverflowError, assign_val, a, -2**127 - 1)
        self.assertRaises(OverflowError, assign_val, a, 2**128)

if __name__ == '__main__':
    unittest.main()

import sys
import unittest
from dynd import nd, ndt

class TestBasics(unittest.TestCase):
    def test_index(self):
        # Test that the __index__ method/nb_index slot
        # in ndobject is working
        a = [1, 2, 3, 4, 5, 6]
        self.assertEqual(a[nd.ndobject(0)], 1)
        self.assertEqual(a[nd.ndobject(1):nd.ndobject(3)], [2, 3])
        self.assertEqual(a[nd.ndobject(-1, ndt.int8)], 6)

    def test_index_errors(self):
        a = [1, 2, 3, 4, 5, 6]
        self.assertRaises(TypeError, lambda x : a[x], nd.ndobject(True))
        self.assertRaises(TypeError, lambda x : a[x], nd.ndobject(3.5))
        self.assertRaises(IndexError, lambda x : a[x], nd.ndobject(10))

    def test_nonzero(self):
        # boolean values
        self.assertFalse(bool(nd.ndobject(False)))
        self.assertTrue(bool(nd.ndobject(True)))
        # integer values
        self.assertFalse(bool(nd.ndobject(0, dtype=ndt.int8)))
        self.assertFalse(bool(nd.ndobject(0, dtype=ndt.uint8)))
        self.assertFalse(bool(nd.ndobject(0, dtype=ndt.int64)))
        self.assertFalse(bool(nd.ndobject(0, dtype=ndt.uint64)))
        self.assertTrue(bool(nd.ndobject(100, dtype=ndt.int8)))
        self.assertTrue(bool(nd.ndobject(100, dtype=ndt.uint8)))
        self.assertTrue(bool(nd.ndobject(100, dtype=ndt.int64)))
        self.assertTrue(bool(nd.ndobject(100, dtype=ndt.uint64)))
        # float values
        self.assertFalse(bool(nd.ndobject(0.0, dtype=ndt.float32)))
        self.assertFalse(bool(nd.ndobject(0.0, dtype=ndt.float64)))
        self.assertTrue(bool(nd.ndobject(100.0, dtype=ndt.float32)))
        self.assertTrue(bool(nd.ndobject(100.0, dtype=ndt.float64)))
        # complex values
        self.assertFalse(bool(nd.ndobject(0.0, dtype=ndt.cfloat32)))
        self.assertFalse(bool(nd.ndobject(0.0, dtype=ndt.cfloat64)))
        self.assertTrue(bool(nd.ndobject(100.0+10.0j, dtype=ndt.cfloat32)))
        self.assertTrue(bool(nd.ndobject(100.0+10.0j, dtype=ndt.cfloat64)))
        # strings
        self.assertFalse(bool(nd.ndobject('')))
        self.assertFalse(bool(nd.ndobject('', ndt.string)))
        self.assertTrue(bool(nd.ndobject(' ')))
        self.assertTrue(bool(nd.ndobject('test', ndt.string)))

    def test_nonzero_errors(self):
        # Non-scalars raise errors like NumPy, because their
        # truth value is ambiguous
        self.assertRaises(ValueError, bool, nd.ndobject([0]))
        self.assertRaises(ValueError, bool, nd.ndobject([1, 2, 3]))
        self.assertRaises(ValueError, bool, nd.ndobject(['abc', 3], dtype='{x:string; y:int32}'))

if __name__ == '__main__':
    unittest.main()

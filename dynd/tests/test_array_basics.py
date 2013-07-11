import sys
import unittest
from dynd import nd, ndt

class TestBasics(unittest.TestCase):
    def test_index(self):
        # Test that the __index__ method/nb_index slot
        # in dynd arrays is working
        a = [1, 2, 3, 4, 5, 6]
        self.assertEqual(a[nd.array(0)], 1)
        self.assertEqual(a[nd.array(1):nd.array(3)], [2, 3])
        self.assertEqual(a[nd.array(-1, ndt.int8)], 6)

    def test_index_errors(self):
        a = [1, 2, 3, 4, 5, 6]
        self.assertRaises(TypeError, lambda x : a[x], nd.array(True))
        self.assertRaises(TypeError, lambda x : a[x], nd.array(3.5))
        self.assertRaises(IndexError, lambda x : a[x], nd.array(10))

    def test_nonzero(self):
        # boolean values
        self.assertFalse(bool(nd.array(False)))
        self.assertTrue(bool(nd.array(True)))
        # integer values
        self.assertFalse(bool(nd.array(0, type=ndt.int8)))
        self.assertFalse(bool(nd.array(0, type=ndt.uint8)))
        self.assertFalse(bool(nd.array(0, type=ndt.int64)))
        self.assertFalse(bool(nd.array(0, type=ndt.uint64)))
        self.assertTrue(bool(nd.array(100, type=ndt.int8)))
        self.assertTrue(bool(nd.array(100, type=ndt.uint8)))
        self.assertTrue(bool(nd.array(100, type=ndt.int64)))
        self.assertTrue(bool(nd.array(100, type=ndt.uint64)))
        # float values
        self.assertFalse(bool(nd.array(0.0, type=ndt.float32)))
        self.assertFalse(bool(nd.array(0.0, type=ndt.float64)))
        self.assertTrue(bool(nd.array(100.0, type=ndt.float32)))
        self.assertTrue(bool(nd.array(100.0, type=ndt.float64)))
        # complex values
        self.assertFalse(bool(nd.array(0.0, type=ndt.cfloat32)))
        self.assertFalse(bool(nd.array(0.0, type=ndt.cfloat64)))
        self.assertTrue(bool(nd.array(100.0+10.0j, type=ndt.cfloat32)))
        self.assertTrue(bool(nd.array(100.0+10.0j, type=ndt.cfloat64)))
        # strings
        self.assertFalse(bool(nd.array('')))
        self.assertFalse(bool(nd.array('', ndt.string)))
        self.assertTrue(bool(nd.array(' ')))
        self.assertTrue(bool(nd.array('test', ndt.string)))

    def test_nonzero_errors(self):
        # Non-scalars raise errors like NumPy, because their
        # truth value is ambiguous
        self.assertRaises(ValueError, bool, nd.array([0]))
        self.assertRaises(ValueError, bool, nd.array([1, 2, 3]))
        self.assertRaises(ValueError, bool, nd.array(['abc', 3], type='{x:string; y:int32}'))

    def test_iter(self):
        # Iteration of a 1D array
        a = nd.array([1, 2, 3])
        self.assertEqual([nd.as_py(x) for x in a], [1, 2, 3])
        # Iteration of a 2D array
        a = nd.array([[1, 2, 3], [4,5,6]])
        self.assertEqual([nd.as_py(x) for x in a], [[1, 2, 3], [4,5,6]])

    def test_iter_fixed_dim(self):
        # Iteration of a 1D array
        a = nd.array([1, 2, 3], type='3, int64')
        self.assertEqual(len(a), 3)
        self.assertEqual([nd.as_py(x) for x in a], [1, 2, 3])
        # Iteration of a 2D array
        a = nd.array([[1, 2, 3], [4,5,6]], type='2, 3, int64')
        self.assertEqual(len(a), 2)
        self.assertEqual([nd.as_py(x) for x in a], [[1, 2, 3], [4,5,6]])

if __name__ == '__main__':
    unittest.main(verbosity=2)

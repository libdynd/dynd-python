import unittest
from dynd import nd, ndt
import math


class TestBasics(unittest.TestCase):
    def test_null_array(self):
        a = nd.array()
        self.assertEqual(str(a), "nd.array()")
        self.assertEqual(repr(a), "nd.array()")
        self.assertRaises(AttributeError, lambda: a.real)
        self.assertRaises(AttributeError, lambda: a.access_flags)
        self.assertRaises(AttributeError, lambda: a.shape)
        self.assertRaises(AttributeError, lambda: a.strides)
        self.assertRaises(AttributeError, lambda: a.is_scalar)

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
        self.assertFalse(bool(nd.array(0.0, type=ndt.complex_float32)))
        self.assertFalse(bool(nd.array(0.0, type=ndt.complex_float64)))
        self.assertTrue(bool(nd.array(100.0+10.0j, type=ndt.complex_float32)))
        self.assertTrue(bool(nd.array(100.0+10.0j, type=ndt.complex_float64)))
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
        self.assertRaises(ValueError, bool, nd.array(['abc', 3], type='{x:string, y:int32}'))

    def test_iter(self):
        # Iteration of a 1D array
        a = nd.array([1, 2, 3])
        self.assertEqual([nd.as_py(x) for x in a], [1, 2, 3])
        # Iteration of a 2D array
        a = nd.array([[1, 2, 3], [4,5,6]])
        self.assertEqual([nd.as_py(x) for x in a], [[1, 2, 3], [4,5,6]])

    def test_iter_fixed_dim(self):
        # Iteration of a 1D array
        a = nd.array([1, 2, 3], type='3 * int64')
        self.assertEqual(len(a), 3)
        self.assertEqual([nd.as_py(x) for x in a], [1, 2, 3])
        # Iteration of a 2D array
        a = nd.array([[1, 2, 3], [4,5,6]], type='2 * 3 * int64')
        self.assertEqual(len(a), 2)
        self.assertEqual([nd.as_py(x) for x in a], [[1, 2, 3], [4,5,6]])

    def test_contiguous(self):
        # Scalar is contiguous
        a = nd.array(3)
        self.assertTrue(nd.is_c_contiguous(a))
        self.assertTrue(nd.is_f_contiguous(a))
        # Default-constructed 1D array
        a = nd.array([1, 3, 5, 2])
        self.assertTrue(nd.is_c_contiguous(a))
        self.assertTrue(nd.is_f_contiguous(a))
        # Sliced 1D array with a step
        a = nd.array([1, 3, 5, 2])[::2]
        self.assertFalse(nd.is_c_contiguous(a))
        self.assertFalse(nd.is_f_contiguous(a))
        a = nd.array([1, 3, 5, 2])[::-1]
        self.assertFalse(nd.is_c_contiguous(a))
        self.assertFalse(nd.is_f_contiguous(a))
        # Default-constructed 2D array
        a = nd.array([[1, 3, 5, 2], [1, 2, 3, 4]])
        self.assertTrue(nd.is_c_contiguous(a))
        self.assertFalse(nd.is_f_contiguous(a))
        # Sliced 2D array with a step
        a = nd.array([[1, 3, 5, 2], [1, 2, 3, 4], [3,4,5,6]])[::2]
        self.assertFalse(nd.is_c_contiguous(a))
        self.assertFalse(nd.is_f_contiguous(a))
        a = nd.array([[1, 3, 5, 2], [1, 2, 3, 4]])[:, ::2]
        self.assertFalse(nd.is_c_contiguous(a))
        self.assertFalse(nd.is_f_contiguous(a))
        a = nd.array([[1, 3, 5, 2], [1, 2, 3, 4]])[::-1, ::-1]
        self.assertFalse(nd.is_c_contiguous(a))
        self.assertFalse(nd.is_f_contiguous(a))

    def test_uint_value_limits(self):
        # Valid maximums
        a = nd.array(0xff, ndt.uint8)
        self.assertEqual(nd.as_py(a), 0xff)
        a = nd.array(0xffff, ndt.uint16)
        self.assertEqual(nd.as_py(a), 0xffff)
        a = nd.array(0xffffffff, ndt.uint32)
        self.assertEqual(nd.as_py(a), 0xffffffff)
        a = nd.array(0xffffffffffffffff, ndt.uint64)
        self.assertEqual(nd.as_py(a), 0xffffffffffffffff)
        # Out of bounds
        self.assertRaises(OverflowError, nd.array, 0x100, ndt.uint8)
        self.assertRaises(OverflowError, nd.array, 0x10000, ndt.uint16)
        self.assertRaises(OverflowError, nd.array, 0x100000000, ndt.uint32)
        self.assertRaises(OverflowError, nd.array, 0x10000000000000000, ndt.uint64)

    def test_inf(self):
        # Validate nd.inf
        self.assertEqual(nd.inf * 2, nd.inf)
        self.assertTrue(nd.inf > 0)
        self.assertTrue(-nd.inf < 0)
        # as an array
        a = nd.array(nd.inf)
        self.assertEqual(nd.as_py(a), nd.inf)
        self.assertEqual(nd.type_of(a), ndt.float64)
        a = nd.array(nd.inf, ndt.float32)
        self.assertEqual(nd.as_py(a), nd.inf)

    def test_nan(self):
        # Validate nd.nan
        self.assertTrue(math.isnan(nd.nan))
        self.assertFalse(nd.nan > 0)
        self.assertFalse(nd.nan < 0)
        self.assertFalse(nd.nan == 0)
        self.assertFalse(nd.nan == nd.nan)
        # as an array
        a = nd.array(nd.nan)
        self.assertTrue(math.isnan(nd.as_py(a)))
        self.assertEqual(nd.type_of(a), ndt.float64)
        a = nd.array(nd.nan, ndt.float32)
        self.assertTrue(math.isnan(nd.as_py(a)))


if __name__ == '__main__':
    unittest.main(verbosity=2)

import sys
import unittest
from dynd import nd, ndt

class TestArraySqueeze(unittest.TestCase):

    def test_squeeze_strided(self):
        # Simple strided array
        a = nd.zeros(1, 1, 3, 1, 2, 1, 1, 1, ndt.int32)
        self.assertEqual(a.shape, (1, 1, 3, 1, 2, 1, 1, 1))
        self.assertEqual(nd.squeeze(a).shape, (3, 1, 2))
        # Strip dimensions from the start
        a = nd.zeros(1, 3, ndt.float32)
        self.assertEqual(a.shape, (1, 3))
        self.assertEqual(nd.squeeze(a).shape, (3,))
        # Strip dimensions from the end
        a = nd.zeros(3, 1, ndt.float32)
        self.assertEqual(a.shape, (3, 1))
        self.assertEqual(nd.squeeze(a).shape, (3,))

    def test_squeeze_var(self):
        # Simple var case (squeeze can see into leading size-1 var dims)
        a = nd.array([[[1], [2,3]]], type='var * var * var * int32')
        self.assertEqual(a.shape, (1, 2, -1))
        self.assertEqual(nd.squeeze(a).shape, (2, -1))
        self.assertEqual(nd.as_py(nd.squeeze(a)), [[1], [2,3]])
        # With an additional fixed 1 dimension at the end
        a = nd.array([[[[1]], [[2], [3]]]])
        self.assertEqual(a.shape, (1, 2, -1, 1))
        self.assertEqual(nd.squeeze(a).shape, (2, -1))
        self.assertEqual(nd.as_py(nd.squeeze(a)), [[1], [2,3]])

    def test_squeeze_axis(self):
        a = nd.zeros(1, 3, 1, 2, 1, ndt.int32)
        self.assertEqual(a.shape, (1, 3, 1, 2, 1))
        self.assertEqual(nd.squeeze(a, axis=0).shape, (3, 1, 2, 1))
        self.assertEqual(nd.squeeze(a, axis=2).shape, (1, 3, 2, 1))
        self.assertEqual(nd.squeeze(a, axis=4).shape, (1, 3, 1, 2))
        self.assertEqual(nd.squeeze(a, axis=-5).shape, (3, 1, 2, 1))
        self.assertEqual(nd.squeeze(a, axis=-3).shape, (1, 3, 2, 1))
        self.assertEqual(nd.squeeze(a, axis=-1).shape, (1, 3, 1, 2))
        self.assertEqual(nd.squeeze(a, axis=(0,2)).shape, (3, 2, 1))
        self.assertEqual(nd.squeeze(a, axis=(0,4)).shape, (3, 1, 2))
        self.assertEqual(nd.squeeze(a, axis=(0,-1)).shape, (3, 1, 2))
        self.assertEqual(nd.squeeze(a, axis=(2,4)).shape, (1, 3, 2))
        self.assertEqual(nd.squeeze(a, axis=(0,2,4)).shape, (3, 2))

    def test_squeeze_errors(self):
        a = nd.zeros(1, 3, 1, 2, 1, ndt.int32)
        # Out of bound axis
        self.assertRaises(IndexError, nd.squeeze, a, axis=-6)
        self.assertRaises(IndexError, nd.squeeze, a, axis=5)
        self.assertRaises(IndexError, nd.squeeze, a, axis=(0, 5))
        self.assertRaises(IndexError, nd.squeeze, a, axis=(2, -6))
        # Dimension of non-one size
        self.assertRaises(IndexError, nd.squeeze, a, axis=1)
        self.assertRaises(IndexError, nd.squeeze, a, axis=(2,3))
        # Axis not an integer
        self.assertRaises(TypeError, nd.squeeze, a, axis=2.0)
        self.assertRaises(TypeError, nd.squeeze, a, axis=(0, 2.0+0j))

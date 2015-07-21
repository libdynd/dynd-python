import sys
import unittest
from dynd import nd, ndt

class TestNumpyCompat(unittest.TestCase):
    """
    Tests to validate interface intended to provide compatibility with the
    NumPy interface.
    """
    def setUp(self):
        if sys.byteorder == 'little':
            self.nonnative = '>'
        else:
            self.nonnative = '<'

    def test_array_attributes(self):
        a = nd.zeros(3, 5, ndt.int32)
        # Property "ndim"
        self.assertEqual(a.ndim, 2)
        # Property "shape"
        self.assertEqual(a.shape, (3, 5))
        # Property "strides"
        self.assertEqual(a.strides, (20, 4))
        # Property "dtype"
        self.assertEqual(a.dtype, ndt.int32)

if __name__ == '__main__':
    unittest.main()

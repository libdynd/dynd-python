import sys
import unittest
from dynd import nd, ndt

class TestArraySetItem(unittest.TestCase):

    def test_strided_dim(self):
        a = nd.empty(100, 'A, int32')
        a[...] = nd.range(100)
        a[0] = 1000
        self.assertEqual(nd.as_py(a[0]), 1000)
        a[1:8:3] = 120
        self.assertEqual(nd.as_py(a[:11]),
                        [1000, 120, 2, 3, 120, 5, 6, 120, 8, 9, 10])
        a[5:2:-1] = [-10, -20, -30]
        self.assertEqual(nd.as_py(a[:11]),
                        [1000, 120, 2, -30, -20, -10, 6, 120, 8, 9, 10])
        a[1] = False
        self.assertEqual(nd.as_py(a[1]), 0)
        a[2] = True
        self.assertEqual(nd.as_py(a[2]), 1)
        a[3] = -10.0
        self.assertEqual(nd.as_py(a[3]), -10)
        a[4] = 101.0 + 0j
        self.assertEqual(nd.as_py(a[4]), 101)

if __name__ == '__main__':
    unittest.main(verbosity=2)

import sys
import unittest
from datetime import date, datetime
from dynd import nd, ndt

class TestPythonList(unittest.TestCase):
    def test_bool(self):
        lst = [True, False, True, True]
        a = nd.array(lst)
        self.assertEqual(nd.dtype_of(a), ndt.bool)
        self.assertEqual(a.shape, (4,))
        self.assertEqual(nd.as_py(a), lst)

        lst = [[True, True], [False, False], [True, False]]
        a = nd.array(lst)
        self.assertEqual(nd.dtype_of(a), ndt.bool)
        self.assertEqual(a.shape, (3,2))
        self.assertEqual(nd.as_py(a), lst)

    def test_int32(self):
        lst = [0, 100, 2000000000, -1000000000]
        a = nd.array(lst)
        self.assertEqual(nd.dtype_of(a), ndt.int32)
        self.assertEqual(a.shape, (4,))
        self.assertEqual(nd.as_py(a), lst)

        lst = [[100, 103, -20], [-30, 0, 1000000]]
        a = nd.array(lst)
        self.assertEqual(nd.dtype_of(a), ndt.int32)
        self.assertEqual(a.shape, (2,3))
        self.assertEqual(nd.as_py(a), lst)

    def test_int64(self):
        lst = [0, 100, 20000000000, -1000000000]
        a = nd.array(lst)
        self.assertEqual(nd.dtype_of(a), ndt.int64)
        self.assertEqual(a.shape, (4,))
        self.assertEqual(nd.as_py(a), lst)

        lst = [[100, 103, -20], [-30, 0, 1000000000000]]
        a = nd.array(lst)
        self.assertEqual(nd.dtype_of(a), ndt.int64)
        self.assertEqual(a.shape, (2,3))
        self.assertEqual(nd.as_py(a), lst)

    def test_float64(self):
        lst = [0, 100.0, 1e25, -1000000000]
        a = nd.array(lst)
        self.assertEqual(nd.dtype_of(a), ndt.float64)
        self.assertEqual(a.shape, (4,))
        self.assertEqual(nd.as_py(a), lst)

        lst = [[100, 103, -20], [-30, 0.0125, 1000000000000]]
        a = nd.array(lst)
        self.assertEqual(nd.dtype_of(a), ndt.float64)
        self.assertEqual(a.shape, (2,3))
        self.assertEqual(nd.as_py(a), lst)

    def test_float64(self):
        lst = [0, 100.0, 1e25, -1000000000+3j]
        a = nd.array(lst)
        self.assertEqual(nd.dtype_of(a), ndt.complex_float64)
        self.assertEqual(a.shape, (4,))
        self.assertEqual(nd.as_py(a), lst)

        lst = [[100, 103j, -20], [-30, 0.0125, 1000000000000]]
        a = nd.array(lst)
        self.assertEqual(nd.dtype_of(a), ndt.complex_float64)
        self.assertEqual(a.shape, (2,3))
        self.assertEqual(nd.as_py(a), lst)

if __name__ == '__main__':
    unittest.main()

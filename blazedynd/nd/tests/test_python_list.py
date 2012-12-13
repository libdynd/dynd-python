import sys
import unittest
from blazedynd import nd

class TestPythonList(unittest.TestCase):
    def test_bool(self):
        lst = [True, False, True, True]
        a = nd.ndobject(lst)
        self.assertEqual(a.dtype, nd.dt.bool)
        self.assertEqual(a.shape, (4,))
        self.assertEqual(a.as_py(), lst)

        lst = [[True, True], [False, False], [True, False]]
        a = nd.ndobject(lst)
        self.assertEqual(a.dtype, nd.dt.bool)
        self.assertEqual(a.shape, (3,2))
        self.assertEqual(a.as_py(), lst)

    def test_int32(self):
        lst = [0, 100, 2000000000, -1000000000]
        a = nd.ndobject(lst)
        self.assertEqual(a.dtype, nd.dt.int32)
        self.assertEqual(a.shape, (4,))
        self.assertEqual(a.as_py(), lst)

        lst = [[100, 103, -20], [-30, 0, 1000000]]
        a = nd.ndobject(lst)
        self.assertEqual(a.dtype, nd.dt.int32)
        self.assertEqual(a.shape, (2,3))
        self.assertEqual(a.as_py(), lst)

    def test_int64(self):
        lst = [0, 100, 20000000000, -1000000000]
        a = nd.ndobject(lst)
        self.assertEqual(a.dtype, nd.dt.int64)
        self.assertEqual(a.shape, (4,))
        self.assertEqual(a.as_py(), lst)

        lst = [[100, 103, -20], [-30, 0, 1000000000000]]
        a = nd.ndobject(lst)
        self.assertEqual(a.dtype, nd.dt.int64)
        self.assertEqual(a.shape, (2,3))
        self.assertEqual(a.as_py(), lst)

    def test_float64(self):
        lst = [0, 100.0, 1e25, -1000000000]
        a = nd.ndobject(lst)
        self.assertEqual(a.dtype, nd.dt.float64)
        self.assertEqual(a.shape, (4,))
        self.assertEqual(a.as_py(), lst)

        lst = [[100, 103, -20], [-30, 0.0125, 1000000000000]]
        a = nd.ndobject(lst)
        self.assertEqual(a.dtype, nd.dt.float64)
        self.assertEqual(a.shape, (2,3))
        self.assertEqual(a.as_py(), lst)

    def test_float64(self):
        lst = [0, 100.0, 1e25, -1000000000+3j]
        a = nd.ndobject(lst)
        self.assertEqual(a.uniform_dtype, nd.dt.complex_float64)
        self.assertEqual(a.shape, (4,))
        self.assertEqual(a.as_py(), lst)

        lst = [[100, 103j, -20], [-30, 0.0125, 1000000000000]]
        a = nd.ndobject(lst)
        self.assertEqual(a.uniform_dtype, nd.dt.complex_float64)
        self.assertEqual(a.shape, (2,3))
        self.assertEqual(a.as_py(), lst)


if __name__ == '__main__':
    unittest.main()

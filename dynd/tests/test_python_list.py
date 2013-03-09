import sys
import unittest
from datetime import date
from dynd import nd, ndt

class TestPythonList(unittest.TestCase):
    def test_bool(self):
        lst = [True, False, True, True]
        a = nd.ndobject(lst)
        self.assertEqual(a.dtype.udtype, ndt.bool)
        self.assertEqual(a.shape, (4,))
        self.assertEqual(nd.as_py(a), lst)

        lst = [[True, True], [False, False], [True, False]]
        a = nd.ndobject(lst)
        self.assertEqual(a.dtype.udtype, ndt.bool)
        self.assertEqual(a.shape, (3,2))
        self.assertEqual(nd.as_py(a), lst)

    def test_int32(self):
        lst = [0, 100, 2000000000, -1000000000]
        a = nd.ndobject(lst)
        self.assertEqual(a.dtype.udtype, ndt.int32)
        self.assertEqual(a.shape, (4,))
        self.assertEqual(nd.as_py(a), lst)

        lst = [[100, 103, -20], [-30, 0, 1000000]]
        a = nd.ndobject(lst)
        self.assertEqual(a.dtype.udtype, ndt.int32)
        self.assertEqual(a.shape, (2,3))
        self.assertEqual(nd.as_py(a), lst)

    def test_int64(self):
        lst = [0, 100, 20000000000, -1000000000]
        a = nd.ndobject(lst)
        self.assertEqual(a.dtype.udtype, ndt.int64)
        self.assertEqual(a.shape, (4,))
        self.assertEqual(nd.as_py(a), lst)

        lst = [[100, 103, -20], [-30, 0, 1000000000000]]
        a = nd.ndobject(lst)
        self.assertEqual(a.dtype.udtype, ndt.int64)
        self.assertEqual(a.shape, (2,3))
        self.assertEqual(nd.as_py(a), lst)

    def test_float64(self):
        lst = [0, 100.0, 1e25, -1000000000]
        a = nd.ndobject(lst)
        self.assertEqual(a.dtype.udtype, ndt.float64)
        self.assertEqual(a.shape, (4,))
        self.assertEqual(nd.as_py(a), lst)

        lst = [[100, 103, -20], [-30, 0.0125, 1000000000000]]
        a = nd.ndobject(lst)
        self.assertEqual(a.dtype.udtype, ndt.float64)
        self.assertEqual(a.shape, (2,3))
        self.assertEqual(nd.as_py(a), lst)

    def test_float64(self):
        lst = [0, 100.0, 1e25, -1000000000+3j]
        a = nd.ndobject(lst)
        self.assertEqual(a.dtype.udtype, ndt.complex_float64)
        self.assertEqual(a.shape, (4,))
        self.assertEqual(nd.as_py(a), lst)

        lst = [[100, 103j, -20], [-30, 0.0125, 1000000000000]]
        a = nd.ndobject(lst)
        self.assertEqual(a.dtype.udtype, ndt.complex_float64)
        self.assertEqual(a.shape, (2,3))
        self.assertEqual(nd.as_py(a), lst)

    def test_date(self):
        lst = [date(2011, 3, 15), date(1933, 12, 25), date(1979, 3, 22)]
        lststr = ['2011-03-15', '1933-12-25', '1979-03-22']
        a = nd.ndobject(lst)
        self.assertEqual(a.dtype.udtype, ndt.date)
        self.assertEqual(a.shape, (3,))
        self.assertEqual(nd.as_py(a), lst)
        self.assertEqual(nd.as_py(a.ucast(ndt.string)), lststr)

if __name__ == '__main__':
    unittest.main()

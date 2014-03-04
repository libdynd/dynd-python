import sys
import unittest
from datetime import date
from dynd import nd, ndt

class TestNDObjectCast(unittest.TestCase):
    def test_broadcast_cast(self):
        a = nd.array(10)
        b = a.cast('3 * int32')
        self.assertRaises(RuntimeError, b.eval)

    def test_strided_to_fixed(self):
        a = nd.array([5,1,2])
        b = a.cast('3 * int32').eval()
        self.assertEqual(nd.as_py(b), [5,1,2])
        self.assertEqual(nd.type_of(b), ndt.type('3 * int32'))

    def test_array_bool(self):
        a = nd.array(True)
        self.assertEqual(bool(a), True)
        self.assertEqual(int(a), 1)
        self.assertEqual(float(a), 1.0)
        self.assertEqual(complex(a), 1.0+0.j)

    def test_array_uint(self):
        a = nd.array(12345, type=ndt.uint32)
        self.assertEqual(bool(a), True)
        self.assertEqual(int(a), 12345)
        self.assertEqual(float(a), 12345.0)
        self.assertEqual(complex(a), 12345.0+0.j)

    def test_array_int(self):
        a = nd.array(-12345)
        self.assertEqual(bool(a), True)
        self.assertEqual(int(a), -12345)
        self.assertEqual(float(a), -12345.0)
        self.assertEqual(complex(a), -12345.0+0.j)

    def test_array_float(self):
        a = nd.array(1.125e-20)
        self.assertEqual(bool(a), True)
        self.assertEqual(float(a), 1.125e-20)
        self.assertEqual(complex(a), 1.125e-20+0.j)

    def test_array_complex(self):
        a = nd.array(1.125e-20+3j)
        self.assertEqual(bool(a), True)
        self.assertEqual(complex(a), 1.125e-20+3j)

if __name__ == '__main__':
    unittest.main()

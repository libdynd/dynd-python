import sys
import unittest
from datetime import date
from dynd import nd, ndt

class TestNDObjectCast(unittest.TestCase):
    def test_broadcast_cast(self):
        a = nd.array(10)
        b = a.cast('3, int32')
        self.assertRaises(RuntimeError, b.eval)

    def test_strided_to_fixed(self):
        a = nd.array([5,1,2])
        b = a.cast('3, int32').eval()
        self.assertEqual(nd.as_py(b), [5,1,2])
        self.assertEqual(nd.type_of(b), ndt.type('3, int32'))

if __name__ == '__main__':
    unittest.main()

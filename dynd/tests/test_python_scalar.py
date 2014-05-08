import sys
import unittest
from dynd import nd, ndt
from datetime import date

if sys.version_info >= (3, 0):
    unicode = str

class TestPythonScalar(unittest.TestCase):
    def test_bool(self):
        # Boolean true/false
        a = nd.array(True)
        self.assertEqual(nd.type_of(a), ndt.bool)
        self.assertEqual(type(nd.as_py(a)), bool)
        self.assertEqual(nd.as_py(a), True)
        a = nd.array(False)
        self.assertEqual(nd.type_of(a), ndt.bool)
        self.assertEqual(type(nd.as_py(a)), bool)
        self.assertEqual(nd.as_py(a), False)

    def test_int(self):
        # Integer that fits in 32 bits
        a = nd.array(10)
        self.assertEqual(nd.type_of(a), ndt.int32)
        self.assertEqual(type(nd.as_py(a)), int)
        self.assertEqual(nd.as_py(a), 10)
        a = nd.array(-2000000000)
        self.assertEqual(nd.type_of(a), ndt.int32)
        self.assertEqual(type(nd.as_py(a)), int)
        self.assertEqual(nd.as_py(a), -2000000000)

        # Integer that requires 64 bits
        a = nd.array(2200000000)
        self.assertEqual(nd.type_of(a), ndt.int64)
        self.assertEqual(nd.as_py(a), 2200000000)
        a = nd.array(-2200000000)
        self.assertEqual(nd.type_of(a), ndt.int64)
        self.assertEqual(nd.as_py(a), -2200000000)

    def test_float(self):
        # Floating point
        a = nd.array(5.125)
        self.assertEqual(nd.type_of(a), ndt.float64)
        self.assertEqual(type(nd.as_py(a)), float)
        self.assertEqual(nd.as_py(a), 5.125)

    def test_complex(self):
        # Complex floating point
        a = nd.array(5.125 - 2.5j)
        self.assertEqual(nd.type_of(a), ndt.complex_float64)
        self.assertEqual(type(nd.as_py(a)), complex)
        self.assertEqual(nd.as_py(a), 5.125 - 2.5j)

    def test_date(self):
        # Date
        a = nd.array(date(2012,12,12))
        self.assertEqual(nd.type_of(a), ndt.date)
        self.assertEqual(type(nd.as_py(a)), date)
        self.assertEqual(nd.as_py(a), date(2012,12,12))

    def test_string(self):
        a = nd.array('abcdef')
        self.assertEqual(nd.type_of(a), ndt.string)
        self.assertEqual(type(nd.as_py(a)), unicode)
        self.assertEqual(nd.as_py(a), u'abcdef')
        a = nd.array(u'abcdef')
        self.assertEqual(nd.type_of(a), ndt.string)
        self.assertEqual(type(nd.as_py(a)), unicode)
        self.assertEqual(nd.as_py(a), u'abcdef')

    def test_utf_encodings(self):
        # Ensure all of the UTF encodings work ok for a basic string
        x = u'\uc548\ub155 hello'
        # UTF-8
        a = nd.array(x)
        a = a.ucast(ndt.make_fixedstring(16, 'utf_8'))
        a = a.eval()
        self.assertEqual(nd.type_of(a), ndt.make_fixedstring(16, 'utf_8'))
        self.assertEqual(type(nd.as_py(a)), unicode)
        self.assertEqual(nd.as_py(a), x)
        # UTF-16
        a = nd.array(x)
        a = a.ucast(ndt.make_fixedstring(8, 'utf_16'))
        a = a.eval()
        self.assertEqual(nd.type_of(a), ndt.make_fixedstring(8, 'utf_16'))
        self.assertEqual(type(nd.as_py(a)), unicode)
        self.assertEqual(nd.as_py(a), x)
        # UTF-32
        a = nd.array(x)
        a = a.ucast(ndt.make_fixedstring(8, 'utf_32'))
        a = a.eval()
        self.assertEqual(nd.type_of(a), ndt.make_fixedstring(8, 'utf_32'))
        self.assertEqual(type(nd.as_py(a)), unicode)
        self.assertEqual(nd.as_py(a), x)

    def test_len(self):
        # Can't get the length of a zero-dimensional dynd array
        a = nd.array(10)
        self.assertRaises(ValueError, len, a)

if __name__ == '__main__':
    unittest.main()

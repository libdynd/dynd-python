import sys
import unittest
from dynd import nd, ndt
from datetime import date

class TestPythonScalar(unittest.TestCase):
    def test_bool(self):
        # Boolean true/false
        a = nd.ndobject(True)
        self.assertEqual(a.dtype, ndt.bool)
        self.assertEqual(type(nd.as_py(a)), bool)
        self.assertEqual(nd.as_py(a), True)
        a = nd.ndobject(False)
        self.assertEqual(a.dtype, ndt.bool)
        self.assertEqual(type(nd.as_py(a)), bool)
        self.assertEqual(nd.as_py(a), False)

    def test_int(self):
        # Integer that fits in 32 bits
        a = nd.ndobject(10)
        self.assertEqual(a.dtype, ndt.int32)
        self.assertEqual(type(nd.as_py(a)), int)
        self.assertEqual(nd.as_py(a), 10)
        a = nd.ndobject(-2000000000)
        self.assertEqual(a.dtype, ndt.int32)
        self.assertEqual(type(nd.as_py(a)), int)
        self.assertEqual(nd.as_py(a), -2000000000)

        # Integer that requires 64 bits
        a = nd.ndobject(2200000000)
        self.assertEqual(a.dtype, ndt.int64)
        self.assertEqual(nd.as_py(a), 2200000000)
        a = nd.ndobject(-2200000000)
        self.assertEqual(a.dtype, ndt.int64)
        self.assertEqual(nd.as_py(a), -2200000000)

    def test_float(self):
        # Floating point
        a = nd.ndobject(5.125)
        self.assertEqual(a.dtype, ndt.float64)
        self.assertEqual(type(nd.as_py(a)), float)
        self.assertEqual(nd.as_py(a), 5.125)

    def test_complex(self):
        # Complex floating point
        a = nd.ndobject(5.125 - 2.5j)
        self.assertEqual(a.dtype, ndt.cfloat64)
        self.assertEqual(type(nd.as_py(a)), complex)
        self.assertEqual(nd.as_py(a), 5.125 - 2.5j)

    def test_date(self):
        # Date
        a = nd.ndobject(date(2012,12,12))
        self.assertEqual(a.dtype, ndt.date)
        self.assertEqual(type(nd.as_py(a)), date)
        self.assertEqual(nd.as_py(a), date(2012,12,12))

    def test_string(self):
        # String/Unicode TODO: Python 3 bytes becomes a bytes<> dtype
        a = nd.ndobject('abcdef')
        self.assertEqual(a.dtype, ndt.make_string_dtype('ascii'))
        self.assertEqual(type(nd.as_py(a)), unicode)
        self.assertEqual(nd.as_py(a), u'abcdef')
        a = nd.ndobject(u'abcdef')
        # Could be UTF 16 or 32 depending on the Python build configuration
        self.assertTrue(a.dtype == ndt.make_string_dtype('ucs_2') or
                    a.dtype == ndt.make_string_dtype('utf_32'))
        self.assertEqual(type(nd.as_py(a)), unicode)
        self.assertEqual(nd.as_py(a), u'abcdef')

    def test_utf_encodings(self):
        # Ensure all of the UTF encodings work ok for a basic string
        x = u'\uc548\ub155 hello'
        # UTF-8
        a = nd.ndobject(x)
        a = a.ucast(ndt.make_fixedstring_dtype(16, 'utf_8'))
        a = a.eval()
        self.assertEqual(a.dtype, ndt.make_fixedstring_dtype(16, 'utf_8'))
        self.assertEqual(type(nd.as_py(a)), unicode)
        self.assertEqual(nd.as_py(a), x)
        # UTF-16
        a = nd.ndobject(x)
        a = a.ucast(ndt.make_fixedstring_dtype(8, 'utf_16'))
        a = a.eval()
        self.assertEqual(a.dtype, ndt.make_fixedstring_dtype(8, 'utf_16'))
        self.assertEqual(type(nd.as_py(a)), unicode)
        self.assertEqual(nd.as_py(a), x)
        # UTF-32
        a = nd.ndobject(x)
        a = a.ucast(ndt.make_fixedstring_dtype(8, 'utf_32'))
        a = a.eval()
        self.assertEqual(a.dtype, ndt.make_fixedstring_dtype(8, 'utf_32'))
        self.assertEqual(type(nd.as_py(a)), unicode)
        self.assertEqual(nd.as_py(a), x)

    def test_len(self):
        # Can't get the length of a zero-dimensional ndobject
        a = nd.ndobject(10)
        self.assertRaises(TypeError, len, a)

if __name__ == '__main__':
    unittest.main()

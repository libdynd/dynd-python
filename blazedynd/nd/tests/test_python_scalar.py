import sys
import unittest
from blazedynd import nd
from datetime import date

class TestPythonScalar(unittest.TestCase):
    def test_bool(self):
        # Boolean true/false
        a = nd.ndobject(True)
        self.assertEqual(a.dtype, nd.dt.bool)
        self.assertEqual(type(a.as_py()), bool)
        self.assertEqual(a.as_py(), True)
        a = nd.ndobject(False)
        self.assertEqual(a.dtype, nd.dt.bool)
        self.assertEqual(type(a.as_py()), bool)
        self.assertEqual(a.as_py(), False)

    def test_int(self):
        # Integer that fits in 32 bits
        a = nd.ndobject(10)
        self.assertEqual(a.dtype, nd.dt.int32)
        self.assertEqual(type(a.as_py()), int)
        self.assertEqual(a.as_py(), 10)
        a = nd.ndobject(-2000000000)
        self.assertEqual(a.dtype, nd.dt.int32)
        self.assertEqual(type(a.as_py()), int)
        self.assertEqual(a.as_py(), -2000000000)

        # Integer that requires 64 bits
        a = nd.ndobject(2200000000)
        self.assertEqual(a.dtype, nd.dt.int64)
        self.assertEqual(a.as_py(), 2200000000)
        a = nd.ndobject(-2200000000)
        self.assertEqual(a.dtype, nd.dt.int64)
        self.assertEqual(a.as_py(), -2200000000)

    def test_float(self):
        # Floating point
        a = nd.ndobject(5.125)
        self.assertEqual(a.dtype, nd.dt.float64)
        self.assertEqual(type(a.as_py()), float)
        self.assertEqual(a.as_py(), 5.125)

    def test_complex(self):
        # Complex floating point
        a = nd.ndobject(5.125 - 2.5j)
        self.assertEqual(a.dtype, nd.dt.cfloat64)
        self.assertEqual(type(a.as_py()), complex)
        self.assertEqual(a.as_py(), 5.125 - 2.5j)

    def test_date(self):
        # Date
        a = nd.ndobject(date(2012,12,12))
        self.assertEqual(a.dtype, nd.dt.date)
        self.assertEqual(type(a.as_py()), date)
        self.assertEqual(a.as_py(), date(2012,12,12))

    def test_string(self):
        # String/Unicode TODO: Python 3 bytes becomes a bytes<> dtype
        a = nd.ndobject('abcdef')
        self.assertEqual(a.dtype, nd.dt.make_string_dtype('ascii'))
        self.assertEqual(type(a.as_py()), unicode)
        self.assertEqual(a.as_py(), u'abcdef')
        a = nd.ndobject(u'abcdef')
        # Could be UTF 16 or 32 depending on the Python build configuration
        self.assertTrue(a.dtype == nd.dt.make_string_dtype('ucs_2') or
                    a.dtype == nd.dt.make_string_dtype('utf_32'))
        self.assertEqual(type(a.as_py()), unicode)
        self.assertEqual(a.as_py(), u'abcdef')

    def test_utf_encodings(self):
        # Ensure all of the UTF encodings work ok for a basic string
        x = u'\uc548\ub155 hello'
        # UTF-8
        a = nd.ndobject(x)
        a = a.cast_scalars(nd.dt.make_fixedstring_dtype('utf_8', 16))
        a = a.vals()
        self.assertEqual(a.dtype, nd.dt.make_fixedstring_dtype('utf_8', 16))
        self.assertEqual(type(a.as_py()), unicode)
        self.assertEqual(a.as_py(), x)
        # UTF-16
        a = nd.ndobject(x)
        a = a.cast_scalars(nd.dt.make_fixedstring_dtype('utf_16', 8))
        a = a.vals()
        self.assertEqual(a.dtype, nd.dt.make_fixedstring_dtype('utf_16', 8))
        self.assertEqual(type(a.as_py()), unicode)
        self.assertEqual(a.as_py(), x)
        # UTF-32
        a = nd.ndobject(x)
        a = a.cast_scalars(nd.dt.make_fixedstring_dtype('utf_32', 8))
        a = a.vals()
        self.assertEqual(a.dtype, nd.dt.make_fixedstring_dtype('utf_32', 8))
        self.assertEqual(type(a.as_py()), unicode)
        self.assertEqual(a.as_py(), x)

    def test_len(self):
        # Can't get the length of a zero-dimensional ndobject
        a = nd.ndobject(10)
        self.assertRaises(TypeError, len, a)

if __name__ == '__main__':
    unittest.main()

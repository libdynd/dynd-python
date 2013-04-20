import sys
import unittest
from dynd import nd, ndt

if sys.version_info >= (3, 0):
    unicode = str

class TestUnicode(unittest.TestCase):
    def test_ndobject_string(self):
        a = nd.ndobject("Testing 1 2 3")
        self.assertEqual(str(a), "Testing 1 2 3")
        self.assertEqual(unicode(a), u"Testing 1 2 3")

    def test_ndobject_unicode(self):
        a = nd.ndobject(u"\uc548\ub155")
        self.assertEqual(unicode(a), u"\uc548\ub155")
        # In Python 2, 'str' is not unicode
        if sys.version_info < (3, 0):
            self.assertRaises(UnicodeEncodeError, str, a)

    def test_ascii_decode_error(self):
        a = nd.ndobject(128, dtype=ndt.uint8).view_scalars("string(1,'A')")
        self.assertRaises(UnicodeDecodeError, a.ucast("string").eval)
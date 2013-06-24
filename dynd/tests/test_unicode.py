import sys
import unittest
from dynd import nd, ndt

if sys.version_info >= (3, 0):
    unicode = str

class TestUnicode(unittest.TestCase):
    def test_array_string(self):
        a = nd.array("Testing 1 2 3")
        self.assertEqual(str(a), "Testing 1 2 3")
        self.assertEqual(unicode(a), u"Testing 1 2 3")

    def test_array_unicode(self):
        a = nd.array(u"\uc548\ub155")
        self.assertEqual(unicode(a), u"\uc548\ub155")
        # In Python 2, 'str' is not unicode
        if sys.version_info < (3, 0):
            self.assertRaises(UnicodeEncodeError, str, a)

    def test_ascii_decode_error(self):
        a = nd.array(128, dtype=ndt.uint8).view_scalars("string(1,'A')")
        self.assertRaises(UnicodeDecodeError, a.ucast("string").eval)
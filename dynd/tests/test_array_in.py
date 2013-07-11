import sys
import unittest
from dynd import nd, ndt

class TestNDObjectIn(unittest.TestCase):
    def test_integers(self):
        a = nd.array([1,2,3,5,6])
        self.assertFalse(0 in a)
        self.assertTrue(1 in a)
        self.assertTrue(2 in a)
        self.assertTrue(3 in a)
        self.assertFalse(4 in a)
        self.assertTrue(5 in a)
        self.assertTrue(6 in a)
        self.assertFalse(7 in a)

    def test_integer_expression(self):
        a = nd.array(['1979-03-22', '1932-12-12', '1999-01-04']).ucast(ndt.date)
        self.assertTrue(1979 in a.year)
        self.assertTrue(1980 not in a.year)
        self.assertTrue(1932 in a.year)
        self.assertTrue(1999 in a.year)
        self.assertFalse(2000 in a.year)

    def test_strings(self):
        a = nd.array(['this', 'is', 'a', 'test'])
        self.assertTrue('this' in a)
        self.assertTrue('is' in a)
        self.assertTrue('a' in a)
        self.assertTrue('test' in a)
        self.assertFalse('' in a)
        self.assertTrue(u'this' in a)
        self.assertTrue(u'is' in a)
        self.assertTrue(u'a' in a)
        self.assertTrue(u'test' in a)
        self.assertFalse(u'' in a)

    def test_strings_different_encoding(self):
        a = nd.array(['this', 'is', 'a', 'test']).ucast(ndt.make_string('utf_8'))
        self.assertTrue('this' in a)
        self.assertTrue('is' in a)
        self.assertTrue('a' in a)
        self.assertTrue('test' in a)
        self.assertFalse('' in a)
        self.assertTrue(u'this' in a)
        self.assertTrue(u'is' in a)
        self.assertTrue(u'a' in a)
        self.assertTrue(u'test' in a)
        self.assertFalse(u'' in a)

if __name__ == '__main__':
    unittest.main()

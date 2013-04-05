import sys
import unittest
from dynd import nd, ndt

class TestBasics(unittest.TestCase):
    def test_index(self):
        # Test that the __index__ method/nb_index slot
        # in ndobject is working
        a = [1, 2, 3, 4, 5, 6]
        self.assertEqual(a[nd.ndobject(0)], 1)
        self.assertEqual(a[nd.ndobject(1):nd.ndobject(3)], [2, 3])
        self.assertEqual(a[nd.ndobject(-1, ndt.int8)], 6)

    def test_index_errors(self):
        a = [1, 2, 3, 4, 5, 6]
        self.assertRaises(TypeError, lambda x : a[x], nd.ndobject(True))
        self.assertRaises(TypeError, lambda x : a[x], nd.ndobject(3.5))
        self.assertRaises(IndexError, lambda x : a[x], nd.ndobject(10))

if __name__ == '__main__':
    unittest.main()

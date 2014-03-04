import sys
import unittest
from datetime import date
from dynd import nd, ndt

# NOTE: These tests are using mockups the follow the blaze interface
#       just enough to be tested. More in-depth testing against blaze
#       is in the blaze repo itself.

class TestConstructFromBlaze(unittest.TestCase):
    def test_blaze_array_in_list(self):
        class Array(object):
            def __init__(self, dshape, value):
                self.dshape = str(dshape)
                self.value = value

            def __int__(self):
                return self.value

            def __float__(self):
                return self.value

        self.assertEqual(type(Array('int32', 1)).__name__, "Array")

        a = nd.array([Array('int32', 3)])
        self.assertEqual(nd.type_of(a), ndt.type('strided * int32'))
        self.assertEqual(nd.as_py(a), [3])

        a = nd.array([Array('float32', 1.25)])
        self.assertEqual(nd.type_of(a), ndt.type('strided * float32'))
        self.assertEqual(nd.as_py(a), [1.25])

if __name__ == '__main__':
    unittest.main()

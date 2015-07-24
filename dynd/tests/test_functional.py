import sys
import unittest

from dynd import nd, ndt

class TestApply(unittest.TestCase):
    def test(self):
        @nd.functional.apply(ndt.callable(ndt.int32, ndt.int32))
        def f(x):
            return x

        self.assertEqual(f.type, ndt.callable(ndt.int32, ndt.int32))
#        self.assertEqual(0, f(0))
 #       self.assertEqual(1, f(1))

if __name__ == '__main__':
    unittest.main()
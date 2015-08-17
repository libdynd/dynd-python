import sys
import unittest

from dynd import nd, ndt

if sys.version_info >= (3,0):
    from ._test_apply3 import TestApply3

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

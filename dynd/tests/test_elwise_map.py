import sys
import unittest
from dynd import nd, ndt

class TestElwiseMap(unittest.TestCase):
    def test_unary_function(self):
        def doubler(dst, src):
            dst[...] = [2 * nd.as_py(x) for x in src]
        # 1D array
        a = nd.arange(5)
        b = nd.elwise_map([a], doubler, ndt.int32)
        self.assertEqual(nd.as_py(b), [0, 2, 4, 6, 8])
        # indexing into the deferred ndobject
        self.assertEqual(nd.as_py(b[1]), 2)
        self.assertEqual(nd.as_py(b[3:]), [6, 8])
        self.assertEqual(nd.as_py(b[1::2]), [2, 6])
        # Modifying 'a' affects 'b'
        a[1:4] = [-1, 10, -2]
        self.assertEqual(nd.as_py(b), [0, -2, 20, -4, 8])

    def test_unary_function_chained(self):
        a = nd.arange(3).cast_udtype(ndt.float32)
        def multiscale(dst, src):
            dst.once = src
            dst.twice = [2 * nd.as_py(x) for x in src]
            dst.thrice = [3 * nd.as_py(x) for x in src]
        b = nd.elwise_map([a], multiscale, nd.dtype('{once: int32; twice: int32; thrice: int32}'))
        self.assertEqual(nd.as_py(b), [{'once':0,'twice':0,'thrice':0},
                        {'once':1,'twice':2,'thrice':3},
                        {'once':2,'twice':4,'thrice':6}])
        self.assertEqual(nd.as_py(b.once), [0, 1, 2])
        self.assertEqual(nd.as_py(b.twice), [0, 2, 4])
        self.assertEqual(nd.as_py(b.thrice), [0, 3, 6])
        # Modifying 'a' affects 'b'
        a[0] = -10
        self.assertEqual(nd.as_py(b.once), [-10, 1, 2])
        self.assertEqual(nd.as_py(b.twice), [-20, 2, 4])
        self.assertEqual(nd.as_py(b.thrice), [-30, 3, 6])

    def test_unary_function_exception(self):
        threshold_val = 10
        def threshold_raise(dst, src):
            for x in src:
                if nd.as_py(x) >= threshold_val:
                    raise ValueError("Bad value %s" % x)
                dst[...] = src
        a = nd.arange(20)
        b = nd.elwise_map([a], threshold_raise, ndt.int32)
        # Should raise when the whole array is evaluated
        self.assertRaises(ValueError, b.eval)
        # If the actual evaluated values are ok, shouldn't raise
        self.assertEqual(nd.as_py(b[5:10]), [5, 6, 7, 8, 9])
        # threshold_raise is a closure, test that it works
        threshold_val = 9
        self.assertRaises(ValueError, b[5:10].eval)

if __name__ == '__main__':
    unittest.main()

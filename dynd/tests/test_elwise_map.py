import sys
import unittest
from dynd import nd, ndt

class TestElwiseMap(unittest.TestCase):
    def test_unary_function(self):
        def doubler(dst, src):
            dst[...] = [2 * nd.as_py(x) for x in src]
        # 1D array
        a = nd.range(5)
        b = nd.elwise_map([a], doubler, ndt.int32)
        self.assertEqual(nd.as_py(b), [0, 2, 4, 6, 8])
        # indexing into the deferred dynd array
        self.assertEqual(nd.as_py(b[1]), 2)
        self.assertEqual(nd.as_py(b[3:]), [6, 8])
        self.assertEqual(nd.as_py(b[1::2]), [2, 6])
        # Modifying 'a' affects 'b'
        a[1:4] = [-1, 10, -2]
        self.assertEqual(nd.as_py(b), [0, -2, 20, -4, 8])

    def test_unary_function_chained(self):
        a = nd.range(3).ucast(ndt.float32)
        def multiscale(dst, src):
            dst.once = src
            dst.twice = [2 * nd.as_py(x) for x in src]
            dst.thrice = [3 * nd.as_py(x) for x in src]
        b = nd.elwise_map([a], multiscale, ndt.type('c{once: int32, twice: int32, thrice: int32}'))
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
        a = nd.range(20)
        b = nd.elwise_map([a], threshold_raise, ndt.int32)
        # Should raise when the whole array is evaluated
        self.assertRaises(ValueError, b.eval)
        # If the actual evaluated values are ok, shouldn't raise
        self.assertEqual(nd.as_py(b[5:10]), [5, 6, 7, 8, 9])
        # threshold_raise is a closure, test that it works
        threshold_val = 9
        self.assertRaises(ValueError, b[5:10].eval)

    def test_simple_computed_column(self):
        def computed_col(dst, src):
            for d, s in zip(dst, src):
                d.fullname = nd.as_py(s.firstname) + ' ' + nd.as_py(s.lastname)
                d.firstname = s.firstname
                d.lastname = s.lastname
                d.country = s.country
        a = nd.parse_json('2 * c{firstname: string, lastname: string, country: string}',
                        """[{"firstname":"Mike", "lastname":"Myers", "country":"Canada"},
                        {"firstname":"Seth", "lastname":"Green", "country":"USA"}]""")
        b = nd.elwise_map([a], computed_col, ndt.type(
                        'c{fullname: string, firstname: string, lastname: string, country: string}'))
        self.assertEqual(nd.as_py(b.fullname), ['Mike Myers', 'Seth Green'])
        self.assertEqual(nd.as_py(b.firstname), ['Mike', 'Seth'])
        self.assertEqual(nd.as_py(b.lastname), ['Myers', 'Green'])
        self.assertEqual(nd.as_py(b.country), ['Canada', 'USA'])

    def test_binary_function(self):
        def multiplier(dst, src0, src1):
            for d, s0, s1 in zip(dst, src0, src1):
                d[...] = nd.as_py(s0) * nd.as_py(s1)
        # 1D array
        a = nd.range(5)
        b = nd.array([1, 3, -2, 4, 12], access='rw')
        c = nd.elwise_map([a,b], multiplier, ndt.int32)
        self.assertEqual(nd.as_py(c), [0, 3, -4, 12, 48])
        # indexing into the deferred dynd array
        self.assertEqual(nd.as_py(c[1]), 3)
        self.assertEqual(nd.as_py(c[3:]), [12, 48])
        self.assertEqual(nd.as_py(c[1::2]), [3, 12])
        # Modifying 'a' or 'b' affects 'c'
        a[1:4] = [-1, 10, -2]
        self.assertEqual(nd.as_py(c), [0, -3, -20, -8, 48])
        b[-1] = 100
        self.assertEqual(nd.as_py(c), [0, -3, -20, -8, 400])

    def test_binary_function_broadcast(self):
        def multiplier(dst, src0, src1):
            for d, s0, s1 in zip(dst, src0, src1):
                d[...] = nd.as_py(s0) * nd.as_py(s1)
        # 1D array
        a = nd.range(5)
        b = nd.array(12).eval_copy(access='readwrite')
        c = nd.elwise_map([a,b], multiplier, ndt.int32)
        self.assertEqual(nd.as_py(c), [0, 12, 24, 36, 48])
        # indexing into the deferred dynd array
        self.assertEqual(nd.as_py(c[1]), 12)
        self.assertEqual(nd.as_py(c[3:]), [36, 48])
        self.assertEqual(nd.as_py(c[1::2]), [12, 36])
        # Modifying 'a' or 'b' affects 'c'
        a[1:4] = [-1, 10, -2]
        self.assertEqual(nd.as_py(c), [0, -12, 120, -24, 48])
        b[...] = 100
        self.assertEqual(nd.as_py(c), [0, -100, 1000, -200, 400])

if __name__ == '__main__':
    unittest.main()

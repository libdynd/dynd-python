import sys
import unittest
from collections import OrderedDict as odict, namedtuple as ntuple
from dynd import nd, ndt

class TestArraySetItem(unittest.TestCase):

    def test_strided_dim(self):
        a = nd.empty(100, ndt.int32)
        a[...] = nd.range(100)
        a[0] = 1000
        self.assertEqual(nd.as_py(a[0]), 1000)
        # Next line causes a crash.
        #a[1:8:3] = 120
        #self.assertEqual(nd.as_py(a[:11]),
        #                [1000, 120, 2, 3, 120, 5, 6, 120, 8, 9, 10])
        #a[5:2:-1] = [-10, -20, -30]
        #self.assertEqual(nd.as_py(a[:11]),
        #                [1000, 120, 2, -30, -20, -10, 6, 120, 8, 9, 10])
        #a[1] = False
        #self.assertEqual(nd.as_py(a[1]), 0)
        #a[2] = True
        #self.assertEqual(nd.as_py(a[2]), 1)
        #a[3] = -10.0
        #self.assertEqual(nd.as_py(a[3]), -10)

#        Should the following even be supported?
#        a[4] = 101.0 + 0j
#        self.assertEqual(nd.as_py(a[4]), 101)

    def test_assign_to_struct(self):
        value = [(8, u'world', 4.5), (16, u'!', 8.75)]
        # Assign list of tuples
        a = nd.empty('2 * { i : int32, msg : string, price : float64 }')
        a[:] = value
        keys = ['i', 'msg', 'price']
        out_val = [{key:val for key, val in zip(keys, val_row)}
                                           for val_row in value]
        self.assertEqual(nd.as_py(a), out_val)
        # Assign iterator of tuples
        a = nd.empty('2 * { i : int32, msg : string, price : float64 }')
        a[:] = iter(value)
        self.assertEqual(nd.as_py(a), out_val)
        # Assign a list of OrderedDicts
        a = nd.empty('2 * { i : int32, msg : string, price : float64 }')
        a[:] = [odict(zip(keys, vals)) for vals in value]
        self.assertEqual(nd.as_py(a), out_val)
        # Assign a list of namedtuples
        named_class = ntuple('named_class', ['i', 'msg', 'price'])
        nt_vals = [named_class(*item) for item in value]
        a = nd.empty('2 * { i : int32, msg : string, price : float64 }')
        a[:] = nt_vals
        self.assertEqual(nd.as_py(a), out_val)

if __name__ == '__main__':
    unittest.main(verbosity=2)

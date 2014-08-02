import sys
import unittest
from dynd import nd, ndt

class TestArrayGetItem(unittest.TestCase):

    def test_strided_dim(self):
        a = nd.empty(100, ndt.int32)
        a[...] = nd.range(100)
        b = list(range(100))
        self.assertEqual(nd.type_of(a), ndt.type('strided * int32'))
        self.assertEqual(nd.type_of(a[...]), ndt.type('strided * int32'))
        self.assertEqual(nd.type_of(a[0]), ndt.int32)
        self.assertEqual(nd.type_of(a[0:1]), ndt.type('strided * int32'))
        self.assertEqual(nd.as_py(a[0]), b[0])
        self.assertEqual(nd.as_py(a[99]), b[99])
        self.assertEqual(nd.as_py(a[-1]), b[-1])
        self.assertEqual(nd.as_py(a[-100]), b[-100])
        self.assertEqual(nd.as_py(a[-101:]), b[-101:])
        self.assertEqual(nd.as_py(a[-5:101:2]), b[-5:101:2])
        self.assertRaises(IndexError, lambda x : x[-101], a)
        self.assertRaises(IndexError, lambda x : x[100], a)

    def test_fixed_dim(self):
        a = nd.empty('100 * int32')
        a[...] = nd.range(100)
        b = list(range(100))
        self.assertEqual(nd.type_of(a), ndt.type('100 * int32'))
        self.assertEqual(nd.type_of(a[...]), ndt.type('100 * int32'))
        self.assertEqual(nd.type_of(a[0]), ndt.int32)
        self.assertEqual(nd.type_of(a[0:1]), ndt.type('strided * int32'))
        self.assertEqual(nd.as_py(a[0]), b[0])
        self.assertEqual(nd.as_py(a[99]), b[99])
        self.assertEqual(nd.as_py(a[-1]), b[-1])
        self.assertEqual(nd.as_py(a[-100]), b[-100])
        self.assertEqual(nd.as_py(a[-101:]), b[-101:])
        self.assertEqual(nd.as_py(a[-5:101:2]), b[-5:101:2])
        self.assertRaises(IndexError, lambda x : x[-101], a)
        self.assertRaises(IndexError, lambda x : x[100], a)

    def test_var_dim(self):
        a = nd.empty('var * int32')
        a[...] = nd.range(100)
        b = list(range(100))
        self.assertEqual(nd.type_of(a), ndt.type('var * int32'))
        self.assertEqual(nd.type_of(a[...]), ndt.type('var * int32'))
        self.assertEqual(nd.type_of(a[:]), ndt.type('strided * int32'))
        self.assertEqual(nd.type_of(a[0]), ndt.int32)
        self.assertEqual(nd.type_of(a[0:1]), ndt.type('strided * int32'))
        self.assertEqual(nd.as_py(a[0]), b[0])
        self.assertEqual(nd.as_py(a[99]), b[99])
        self.assertEqual(nd.as_py(a[-1]), b[-1])
        self.assertEqual(nd.as_py(a[-100]), b[-100])
        self.assertEqual(nd.as_py(a[-101:]), b[-101:])
        self.assertEqual(nd.as_py(a[-5:101:2]), b[-5:101:2])
        self.assertRaises(IndexError, lambda x : x[-101], a)
        self.assertRaises(IndexError, lambda x : x[100], a)

    def test_struct(self):
        a = nd.parse_json('{x:int32, y:string, z:float32}',
                        '{"x":20, "y":"testing one two three", "z":-3.25}')
        self.assertEqual(nd.type_of(a), ndt.type('{x:int32, y:string, z:float32}'))
        self.assertEqual(nd.type_of(a[...]), ndt.type('{x:int32, y:string, z:float32}'))
        self.assertEqual(nd.type_of(a[0]), ndt.int32)
        self.assertEqual(nd.type_of(a[1]), ndt.string)
        self.assertEqual(nd.type_of(a[2]), ndt.float32)
        self.assertEqual(nd.type_of(a[-3]), ndt.int32)
        self.assertEqual(nd.type_of(a[-2]), ndt.string)
        self.assertEqual(nd.type_of(a[-1]), ndt.float32)
        self.assertEqual(nd.type_of(a[1:]), ndt.make_struct([ndt.string, ndt.float32], ['y', 'z']))
        self.assertEqual(nd.type_of(a[::-2]), ndt.make_struct([ndt.float32, ndt.int32], ['z', 'x']))
        self.assertEqual(nd.as_py(a[0]), 20)
        self.assertEqual(nd.as_py(a[1]), "testing one two three")
        self.assertEqual(nd.as_py(a[2]), -3.25)
        self.assertEqual(nd.as_py(a[1:]), {'y':'testing one two three', 'z':-3.25})
        self.assertEqual(nd.as_py(a[::-2]), {'x':20, 'z':-3.25})

    def test_nd_array_int(self):
        a = nd.range(3)
        self.assertEqual(nd.as_py(a[a[0]]), 0)
        self.assertEqual(nd.as_py(a[a[0] + 1]), 1)


if __name__ == '__main__':
    unittest.main()

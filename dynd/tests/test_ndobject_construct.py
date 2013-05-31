import sys
import unittest
from datetime import date
from dynd import nd, ndt

class TestStringConstruct(unittest.TestCase):
    def test_string(self):
        a = nd.ndobject('abc', dtype=ndt.string)
        self.assertEqual(a.dtype, ndt.string)
        a = nd.ndobject('abc', udtype=ndt.string)
        self.assertEqual(a.dtype, ndt.string)

    def test_string_array(self):
        a = nd.ndobject(['this', 'is', 'a', 'test'],
                        udtype=ndt.string)
        self.assertEqual(a.dtype, nd.dtype('N, string'))
        self.assertEqual(nd.as_py(a), ['this', 'is', 'a', 'test'])

        a = nd.ndobject(['this', 'is', 'a', 'test'],
                        udtype='string("U16")')
        self.assertEqual(a.dtype, nd.dtype('N, string("U16")'))
        self.assertEqual(nd.as_py(a), ['this', 'is', 'a', 'test'])

    def test_fixedstring_array(self):
        a = nd.ndobject(['a', 'b', 'c'],
                        udtype='string(1,"A")')
        self.assertEqual(a[0].dtype.type_id, 'fixedstring')
        self.assertEqual(a[0].dtype.data_size, 1)
        self.assertEqual(nd.as_py(a), ['a', 'b', 'c'])


class TestStructConstruct(unittest.TestCase):
    def test_single_struct(self):
        a = nd.ndobject([12, 'test', True], dtype='{x:int32; y:string; z:bool}')
        self.assertEqual(a.dtype, nd.dtype('{x:int32; y:string; z:bool}'))
        self.assertEqual(nd.as_py(a[0]), 12)
        self.assertEqual(nd.as_py(a[1]), 'test')
        self.assertEqual(nd.as_py(a[2]), True)

        a = nd.ndobject({'x':12, 'y':'test', 'z':True}, dtype='{x:int32; y:string; z:bool}')
        self.assertEqual(a.dtype, nd.dtype('{x:int32; y:string; z:bool}'))
        self.assertEqual(nd.as_py(a[0]), 12)
        self.assertEqual(nd.as_py(a[1]), 'test')
        self.assertEqual(nd.as_py(a[2]), True)

    def test_nested_struct(self):
        a = nd.ndobject([[1,2], ['test', 3.5], [3j]],
                    dtype='{x: 2, int16; y: {a: string; b: float64}; z: 1, cfloat32}')
        self.assertEqual(nd.as_py(a.x), [1, 2])
        self.assertEqual(nd.as_py(a.y.a), 'test')
        self.assertEqual(nd.as_py(a.y.b), 3.5)
        self.assertEqual(nd.as_py(a.z), [3j])

        a = nd.ndobject({'x':[1,2], 'y':{'a':'test', 'b':3.5}, 'z':[3j]},
                    dtype='{x: 2, int16; y: {a: string; b: float64}; z: 1, cfloat32}')
        self.assertEqual(nd.as_py(a.x), [1, 2])
        self.assertEqual(nd.as_py(a.y.a), 'test')
        self.assertEqual(nd.as_py(a.y.b), 3.5)
        self.assertEqual(nd.as_py(a.z), [3j])

    def test_single_struct_array(self):
        a = nd.ndobject([(0,0), (3,5), (12,10)], udtype='{x:int32; y:int32}')
        self.assertEqual(a.dtype, nd.dtype('N, {x:int32; y:int32}'))
        self.assertEqual(nd.as_py(a.x), [0, 3, 12])
        self.assertEqual(nd.as_py(a.y), [0, 5, 10])

        a = nd.ndobject([{'x':0,'y':0}, {'x':3,'y':5}, {'x':12,'y':10}],
                    udtype='{x:int32; y:int32}')
        self.assertEqual(a.dtype, nd.dtype('N, {x:int32; y:int32}'))
        self.assertEqual(nd.as_py(a.x), [0, 3, 12])
        self.assertEqual(nd.as_py(a.y), [0, 5, 10])

        a = nd.ndobject([[(3, 'X')], [(10, 'L'), (12, 'M')]],
                        udtype='{count:int32; size:string(1,"A")}')
        self.assertEqual(a.dtype, nd.dtype('N, var, {count:int32; size:string(1,"A")}'))
        self.assertEqual(nd.as_py(a.count), [[3], [10, 12]])
        self.assertEqual(nd.as_py(a.size), [['X'], ['L', 'M']])

        a = nd.ndobject([[{'count':3, 'size':'X'}],
                        [{'count':10, 'size':'L'}, {'count':12, 'size':'M'}]],
                        udtype='{count:int32; size:string(1,"A")}')
        self.assertEqual(a.dtype, nd.dtype('N, var, {count:int32; size:string(1,"A")}'))
        self.assertEqual(nd.as_py(a.count), [[3], [10, 12]])
        self.assertEqual(nd.as_py(a.size), [['X'], ['L', 'M']])

    def test_nested_struct_array(self):
        a = nd.ndobject([((0,1),0), ((2,2),5), ((100,10),10)],
                    udtype='{x:{a:int16; b:int16}; y:int32}')
        self.assertEqual(a.dtype, nd.dtype('N, {x:{a:int16; b:int16}; y:int32}'))
        self.assertEqual(nd.as_py(a.x.a), [0, 2, 100])
        self.assertEqual(nd.as_py(a.x.b), [1, 2, 10])
        self.assertEqual(nd.as_py(a.y), [0, 5, 10])

        a = nd.ndobject([{'x':{'a':0,'b':1},'y':0},
                        {'x':{'a':2,'b':2},'y':5},
                        {'x':{'a':100,'b':10},'y':10}],
                    udtype='{x:{a:int16; b:int16}; y:int32}')
        self.assertEqual(a.dtype, nd.dtype('N, {x:{a:int16; b:int16}; y:int32}'))
        self.assertEqual(nd.as_py(a.x.a), [0, 2, 100])
        self.assertEqual(nd.as_py(a.x.b), [1, 2, 10])
        self.assertEqual(nd.as_py(a.y), [0, 5, 10])

        a = nd.ndobject([[(3, ('X', 10))], [(10, ('L', 7)), (12, ('M', 5))]],
                        udtype='{count:int32; size:{name:string(1,"A"); id: int8}}')
        self.assertEqual(a.dtype,
                    nd.dtype('N, var, {count:int32; size:{name:string(1,"A"); id: int8}}'))
        self.assertEqual(nd.as_py(a.count), [[3], [10, 12]])
        self.assertEqual(nd.as_py(a.size.name), [['X'], ['L', 'M']])
        self.assertEqual(nd.as_py(a.size.id), [[10], [7, 5]])

    def test_missing_field(self):
        self.assertRaises(RuntimeError, nd.ndobject,
                        [0, 1], udtype='{x:int32; y:int32; z:int32}')
        self.assertRaises(RuntimeError, nd.ndobject,
                        {'x':0, 'z':1}, udtype='{x:int32; y:int32; z:int32}')

    def test_extra_field(self):
        self.assertRaises(RuntimeError, nd.ndobject,
                        [0, 1, 2, 3], udtype='{x:int32; y:int32; z:int32}')
        self.assertRaises(RuntimeError, nd.ndobject,
                        {'x':0,'y':1,'z':2,'w':3}, udtype='{x:int32; y:int32; z:int32}')

if __name__ == '__main__':
    unittest.main()

import sys
import unittest
from datetime import date
from dynd import nd, ndt
import numpy as np

class TestCopyFromPy(unittest.TestCase):
    def test_bool(self):
        a = nd.empty('var * bool')
        a[...] = [True, False, 1, 0, 'true', 'false', 'on', 'off']
        self.assertEqual(nd.as_py(a), [True, False] * 4)

class TestCopyFromNumPy(unittest.TestCase):
    def test_simple_strided(self):
        a = nd.empty('3 * int32')
        a[...] = np.int64(1)
        self.assertEqual(nd.as_py(a), [1] * 3)
        a[...] = np.array(2.0)
        self.assertEqual(nd.as_py(a), [2] * 3)
        a[...] = np.array([3], dtype=np.int8)
        self.assertEqual(nd.as_py(a), [3] * 3)
        a[...] = np.array([1, 2, 3])
        self.assertEqual(nd.as_py(a), [1, 2, 3])

    def test_simple_var(self):
        a = nd.empty('var * int32')
        a[...] = np.int64(1)
        self.assertEqual(nd.as_py(a), [1])

        a = nd.empty('var * int32')
        a[...] = np.array(2.0)
        self.assertEqual(nd.as_py(a), [2])

        a = nd.empty('var * int32')
        a[...] = np.array([3], dtype=np.int8)
        self.assertEqual(nd.as_py(a), [3])

        a = nd.empty('var * int32')
        a[...] = np.array([1, 2, 3])
        self.assertEqual(nd.as_py(a), [1, 2, 3])
        a[...] = np.array([4])
        self.assertEqual(nd.as_py(a), [4] * 3)

    def test_object_arr(self):
        a = nd.empty('3 * int')
        a[...] = np.array([1, 2, 3.0], dtype=object)
        self.assertEqual(nd.as_py(a), [1, 2, 3])

        a = nd.empty('3 * string')
        a[...] = np.array(['testing', 'one', u'two'], dtype=object)
        self.assertEqual(nd.as_py(a), ['testing', 'one', 'two'])

        a = nd.empty('3 * string')
        a[...] = np.array(['broadcast_string'], dtype=object)
        self.assertEqual(nd.as_py(a), ['broadcast_string'] * 3)

        a = nd.empty('3 * string')
        a[...] = np.array('testing', dtype=object)
        self.assertEqual(nd.as_py(a), ['testing'] * 3)

    def test_object_in_struct_arr(self):
        a = nd.empty('3 * {x: int, y: string}')
        a[...] = np.array([(1, 'test'), (2, u'one'), (3.0, 'two')],
                          dtype=[('x', np.int64), ('y', object)])
        self.assertEqual(nd.as_py(a, tuple=True),
                         [(1, 'test'), (2, 'one'), (3, 'two')])

        a = nd.empty('3 * {x: int, y: string}')
        a[...] = np.array([('opposite', 4)],
                          dtype=[('y', object), ('x', np.int64)])
        self.assertEqual(nd.as_py(a, tuple=True),
                         [(4, 'opposite')] * 3)

        a = nd.empty('var * {x: int, y: string}')
        a[...] = np.array([(1, 'test'), (2, u'one'), (3.0, 'two')],
                          dtype=[('x', object), ('y', object)])
        self.assertEqual(nd.as_py(a, tuple=True),
                         [(1, 'test'), (2, 'one'), (3, 'two')])

class TestStructCopy(unittest.TestCase):
    def test_single_struct(self):
        a = nd.empty('{x:int32, y:string, z:bool}')
        a[...] = [3, 'test', False]
        self.assertEqual(nd.as_py(a.x), 3)
        self.assertEqual(nd.as_py(a.y), 'test')
        self.assertEqual(nd.as_py(a.z), False)

        a = nd.empty('{x:int32, y:string, z:bool}')
        a[...] = {'x':10, 'y':'testing', 'z':True}
        self.assertEqual(nd.as_py(a.x), 10)
        self.assertEqual(nd.as_py(a.y), 'testing')
        self.assertEqual(nd.as_py(a.z), True)

    def test_nested_struct(self):
        a = nd.empty('{x: 2 * int16, y: {a: string, b: float64}, z: 1 * complex[float32]}')
        a[...] = [[1,2], ['test', 3.5], [3j]]
        self.assertEqual(nd.as_py(a.x), [1, 2])
        self.assertEqual(nd.as_py(a.y.a), 'test')
        self.assertEqual(nd.as_py(a.y.b), 3.5)
        self.assertEqual(nd.as_py(a.z), [3j])

        a = nd.empty('{x: 2 * int16, y: {a: string, b: float64}, z: 1 * complex[float32]}')
        a[...] = {'x':[1,2], 'y':{'a':'test', 'b':3.5}, 'z':[3j]}
        self.assertEqual(nd.as_py(a.x), [1, 2])
        self.assertEqual(nd.as_py(a.y.a), 'test')
        self.assertEqual(nd.as_py(a.y.b), 3.5)
        self.assertEqual(nd.as_py(a.z), [3j])

    def test_single_struct_array(self):
        a = nd.empty('3 * c{x:int32, y:int32}')
        a[...] = [(0,0), (3,5), (12,10)]
        self.assertEqual(nd.as_py(a.x), [0, 3, 12])
        self.assertEqual(nd.as_py(a.y), [0, 5, 10])

        a[...] = [{'x':1,'y':2}, {'x':4,'y':7}, {'x':14,'y':190}]
        self.assertEqual(nd.as_py(a.x), [1, 4, 14])
        self.assertEqual(nd.as_py(a.y), [2, 7, 190])

        a = nd.empty('2 * var * c{count:int32, size:string[1,"A"]}')
        a[...] = [[(3, 'X')], [(10, 'L'), (12, 'M')]]
        self.assertEqual(nd.as_py(a.count), [[3], [10, 12]])
        self.assertEqual(nd.as_py(a.size), [['X'], ['L', 'M']])

        a[...] = [[{'count':6, 'size':'M'}],
                        [{'count':3, 'size':'F'}, {'count':16, 'size':'D'}]]
        self.assertEqual(nd.as_py(a.count), [[6], [3, 16]])
        self.assertEqual(nd.as_py(a.size), [['M'], ['F', 'D']])

        a[...] = {'count':1, 'size':'Z'}
        self.assertEqual(nd.as_py(a.count), [[1], [1, 1]])
        self.assertEqual(nd.as_py(a.size), [['Z'], ['Z', 'Z']])

        a[...] = [[(10, 'A')], [(5, 'B')]]
        self.assertEqual(nd.as_py(a.count), [[10], [5, 5]])
        self.assertEqual(nd.as_py(a.size), [['A'], ['B', 'B']])

    def test_nested_struct_array(self):
        a = nd.empty('3 * c{x:c{a:int16, b:int16}, y:int32}')
        a[...] = [((0,1),0), ((2,2),5), ((100,10),10)]
        self.assertEqual(nd.as_py(a.x.a), [0, 2, 100])
        self.assertEqual(nd.as_py(a.x.b), [1, 2, 10])
        self.assertEqual(nd.as_py(a.y), [0, 5, 10])

        a[...] = [{'x':{'a':1,'b':2},'y':5},
                  {'x':{'a':3,'b':6},'y':7},
                  {'x':{'a':1001,'b':110},'y':110}]
        self.assertEqual(nd.as_py(a.x.a), [1, 3, 1001])
        self.assertEqual(nd.as_py(a.x.b), [2, 6, 110])
        self.assertEqual(nd.as_py(a.y), [5, 7, 110])

        a = nd.empty('2 * var * c{count:int32, size:c{name:string[1,"A"], id: int8}}')
        a[...] = [[(3, ('X', 10))], [(10, ('L', 7)), (12, ('M', 5))]]
        self.assertEqual(nd.as_py(a.count), [[3], [10, 12]])
        self.assertEqual(nd.as_py(a.size.name), [['X'], ['L', 'M']])
        self.assertEqual(nd.as_py(a.size.id), [[10], [7, 5]])

    def test_missing_field(self):
        a = nd.empty('{x:int32, y:int32, z:int32}')
        def assign(x, y):
            x[...] = y
        self.assertRaises(nd.BroadcastError, assign, a, [0, 1])
        self.assertRaises(nd.BroadcastError, assign, a, {'x':0, 'z':1})

    def test_extra_field(self):
        a = nd.empty('{x:int32, y:int32, z:int32}')
        def assign(x, y):
            x[...] = y
        self.assertRaises(nd.BroadcastError, assign, a, [0, 1, 2, 3])
        self.assertRaises(nd.BroadcastError, assign, a, {'x':0,'y':1,'z':2,'w':3})

class TestIteratorAssign(unittest.TestCase):
    def test_simple_var_dim(self):
        # Assign to a var dim from a generator
        a = nd.empty('var * int32')
        a[...] = (x + 2 for x in range(10))
        self.assertEqual(len(a), 10)
        self.assertEqual(nd.as_py(a), [x + 2 for x in range(10)])
        # If we assign from a generator with one element, it broadcasts
        a[...] = (x + 3 for x in range(5,6))
        self.assertEqual(len(a), 10)
        self.assertEqual(nd.as_py(a), [8]*10)

        def assign(x, y):
            x[...] = y
        # If we assign from a generator with too few elements, it errors
        self.assertRaises(nd.BroadcastError, assign, a,
                        (x + 2 for x in range(9)))
        # If we assign from a generator with too many elements, it errors
        self.assertRaises(nd.BroadcastError, assign, a,
                        (x + 2 for x in range(11)))

    def test_simple_strided_dim(self):
        # Assign to a strided dim from a generator
        a = nd.empty(10, ndt.int32)
        a[...] = (x + 2 for x in range(10))
        self.assertEqual(len(a), 10)
        self.assertEqual(nd.as_py(a), [x + 2 for x in range(10)])
        # If we assign from a generator with one element, it broadcasts
        a[...] = (x + 3 for x in range(5,6))
        self.assertEqual(len(a), 10)
        self.assertEqual(nd.as_py(a), [8]*10)

        def assign(x, y):
            x[...] = y
        # If we assign from a generator with too few elements, it errors
        self.assertRaises(nd.BroadcastError, assign, a,
                        (x + 2 for x in range(9)))
        # If we assign from a generator with too many elements, it errors
        self.assertRaises(nd.BroadcastError, assign, a,
                        (x + 2 for x in range(11)))

    def test_simple_fixed_dim(self):
        # Assign to a strided dim from a generator
        a = nd.empty(10, ndt.int32)
        a[...] = (x + 2 for x in range(10))
        self.assertEqual(len(a), 10)
        self.assertEqual(nd.as_py(a), [x + 2 for x in range(10)])
        # If we assign from a generator with one element, it broadcasts
        a[...] = (x + 3 for x in range(5,6))
        self.assertEqual(len(a), 10)
        self.assertEqual(nd.as_py(a), [8]*10)

        def assign(x, y):
            x[...] = y
        # If we assign from a generator with too few elements, it errors
        self.assertRaises(nd.BroadcastError, assign, a,
                        (x + 2 for x in range(9)))
        # If we assign from a generator with too many elements, it errors
        self.assertRaises(nd.BroadcastError, assign, a,
                        (x + 2 for x in range(11)))

class TestStringCopy(unittest.TestCase):
    def test_string_assign_to_slice(self):
        a = nd.array(['a', 'b', 'c', 'd', 'e'], 'string[8]', access='rw')
        a[:3] = 'test'
        self.assertEqual(nd.as_py(a), ['test', 'test', 'test', 'd', 'e'])


if __name__ == '__main__':
    unittest.main()

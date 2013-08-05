import sys
import unittest
from datetime import date
from dynd import nd, ndt

class TestArrayConstructor(unittest.TestCase):
    # Always constructs a new array
    def test_simple(self):
        a = nd.array([1, 2, 3], access='rw')
        self.assertEqual(nd.type_of(a), ndt.type('M, int32'))
        # Modifying 'a' shouldn't affect 'b', because it's a copy
        b = nd.array(a)
        a[1] = 10
        self.assertEqual(nd.as_py(b), [1, 2, 3])

    def test_access_from_pyobject(self):
        a = nd.array([1, 2, 3])
        self.assertEqual(a.access_flags, 'immutable')
        a = nd.array([1, 2, 3], access='immutable')
        self.assertEqual(a.access_flags, 'immutable')
        a = nd.array([1, 2, 3], access='readonly')
        self.assertEqual(a.access_flags, 'immutable')
        a = nd.array([1, 2, 3], access='r')
        self.assertEqual(a.access_flags, 'immutable')
        a = nd.array([1, 2, 3], access='readwrite')
        self.assertEqual(a.access_flags, 'readwrite')
        a = nd.array([1, 2, 3], access='rw')
        self.assertEqual(a.access_flags, 'readwrite')

    def test_access_from_immutable_array(self):
        # `a` is an immutable array
        a = nd.array([1, 2, 3])
        b = nd.array(a)
        self.assertEqual(b.access_flags, 'immutable')
        b = nd.array(a, access='immutable')
        self.assertEqual(b.access_flags, 'immutable')
        b = nd.array(a, access='readonly')
        self.assertEqual(b.access_flags, 'immutable')
        b = nd.array(a, access='r')
        self.assertEqual(b.access_flags, 'immutable')
        b = nd.array(a, access='readwrite')
        self.assertEqual(b.access_flags, 'readwrite')
        b = nd.array(a, access='rw')
        self.assertEqual(b.access_flags, 'readwrite')

    def test_access_from_readwrite_array(self):
        # `a` is a readwrite array
        a = nd.array([1, 2, 3], access='rw')
        b = nd.array(a)
        self.assertEqual(b.access_flags, 'immutable')
        b = nd.array(a, access='immutable')
        self.assertEqual(b.access_flags, 'immutable')
        b = nd.array(a, access='readonly')
        self.assertEqual(b.access_flags, 'immutable')
        b = nd.array(a, access='r')
        self.assertEqual(b.access_flags, 'immutable')
        b = nd.array(a, access='readwrite')
        self.assertEqual(b.access_flags, 'readwrite')
        b = nd.array(a, access='rw')
        self.assertEqual(b.access_flags, 'readwrite')

class TestViewConstructor(unittest.TestCase):
    # Always constructs a view
    def test_simple(self):
        a = nd.array([1, 2, 3], access='rw')
        # Modifying 'a' should affect 'b', because it's a view
        b = nd.view(a)
        a[1] = 10
        self.assertEqual(nd.as_py(b), [1, 10, 3])
        # Can't construct a view of a python list
        self.assertRaises(RuntimeError, nd.view, [1, 2, 3])

    def test_access_from_immutable_array(self):
        # `a` is an immutable array
        a = nd.array([1, 2, 3])
        b = nd.view(a)
        self.assertEqual(b.access_flags, 'immutable')
        b = nd.view(a, access='immutable')
        self.assertEqual(b.access_flags, 'immutable')
        b = nd.view(a, access='readonly')
        self.assertEqual(b.access_flags, 'immutable')
        b = nd.view(a, access='r')
        self.assertEqual(b.access_flags, 'immutable')
        # Can't create a readwrite view from a readonly array
        self.assertRaises(RuntimeError, nd.view, a, access='readwrite')
        self.assertRaises(RuntimeError, nd.view, a, access='rw')

    def test_access_from_readwrite_array(self):
        # `a` is a readwrite array
        a = nd.array([1, 2, 3], access='rw')
        b = nd.view(a)
        self.assertEqual(b.access_flags, 'readwrite')
        # Can't create an immutable view of a readwrite array
        self.assertRaises(RuntimeError, nd.view, a, access='immutable')
        b = nd.view(a, access='readonly')
        self.assertEqual(b.access_flags, 'readonly')
        b = nd.view(a, access='r')
        self.assertEqual(b.access_flags, 'readonly')
        b = nd.view(a, access='readwrite')
        self.assertEqual(b.access_flags, 'readwrite')
        b = nd.view(a, access='rw')
        self.assertEqual(b.access_flags, 'readwrite')

class TestAsArrayConstructor(unittest.TestCase):
    # Constructs a view if possible, otherwise a copy
    def test_simple(self):
        a = nd.asarray([1, 2, 3], access='rw')
        self.assertEqual(nd.type_of(a), ndt.type('M, int32'))

        # Modifying 'a' should affect 'b', because it's a view
        b = nd.asarray(a)
        self.assertEqual(nd.as_py(b), [1, 2, 3])
        a[1] = 10
        self.assertEqual(nd.as_py(b), [1, 10, 3])

        # Can take a readonly view, but still modify the original
        b = nd.asarray(a, access='r')
        self.assertEqual(nd.as_py(b), [1, 10, 3])
        a[1] = 20
        self.assertEqual(nd.as_py(b), [1, 20, 3])
        # The readonly view we took can't be written to
        def assign_at(x, i, y):
            x[i] = y
        self.assertRaises(RuntimeError, assign_at, b, 1, 30)

        # Asking for immutable makes a copy instead of a view
        b = nd.asarray(a, access='immutable')
        self.assertEqual(nd.as_py(b), [1, 20, 3])
        a[1] = 40
        self.assertEqual(nd.as_py(b), [1, 20, 3])

        # Asking for immutable from a non-immutable
        # readonly array makes a copy
        aprime = nd.asarray(a, access='r')
        b = nd.asarray(aprime, access='immutable')
        self.assertEqual(nd.as_py(aprime), [1, 40, 3])
        self.assertEqual(nd.as_py(b), [1, 40, 3])
        a[1] = 50
        self.assertEqual(nd.as_py(aprime), [1, 50, 3])
        self.assertEqual(nd.as_py(b), [1, 40, 3])

    def test_access_from_pyobject(self):
        a = nd.asarray([1, 2, 3])
        self.assertEqual(a.access_flags, 'immutable')
        a = nd.asarray([1, 2, 3], access='immutable')
        self.assertEqual(a.access_flags, 'immutable')
        a = nd.asarray([1, 2, 3], access='readonly')
        self.assertEqual(a.access_flags, 'immutable')
        a = nd.asarray([1, 2, 3], access='r')
        self.assertEqual(a.access_flags, 'immutable')
        a = nd.asarray([1, 2, 3], access='readwrite')
        self.assertEqual(a.access_flags, 'readwrite')
        a = nd.asarray([1, 2, 3], access='rw')
        self.assertEqual(a.access_flags, 'readwrite')

    def test_access_from_immutable_array(self):
        # `a` is an immutable array
        a = nd.array([1, 2, 3])
        b = nd.asarray(a)
        self.assertEqual(b.access_flags, 'immutable')
        b = nd.asarray(a, access='immutable')
        self.assertEqual(b.access_flags, 'immutable')
        b = nd.asarray(a, access='readonly')
        self.assertEqual(b.access_flags, 'immutable')
        b = nd.asarray(a, access='r')
        self.assertEqual(b.access_flags, 'immutable')
        b = nd.asarray(a, access='readwrite')
        self.assertEqual(b.access_flags, 'readwrite')
        b = nd.asarray(a, access='rw')
        self.assertEqual(b.access_flags, 'readwrite')

    def test_access_from_readwrite_array(self):
        # `a` is a readwrite array
        a = nd.array([1, 2, 3], access='rw')
        b = nd.asarray(a)
        self.assertEqual(b.access_flags, 'readwrite')
        b = nd.asarray(a, access='immutable')
        self.assertEqual(b.access_flags, 'immutable')
        b = nd.asarray(a, access='readonly')
        self.assertEqual(b.access_flags, 'readonly')
        b = nd.asarray(a, access='r')
        self.assertEqual(b.access_flags, 'readonly')
        b = nd.asarray(a, access='readwrite')
        self.assertEqual(b.access_flags, 'readwrite')
        b = nd.asarray(a, access='rw')
        self.assertEqual(b.access_flags, 'readwrite')

class TestStringConstruct(unittest.TestCase):
    def test_empty_array(self):
        # Empty arrays default to float64
        a = nd.array([])
        self.assertEqual(nd.type_of(a), ndt.type('M, float64'))
        self.assertEqual(a.shape, (0,))
        a = nd.array([[], [], []])
        self.assertEqual(nd.type_of(a), ndt.type('M, N, float64'))
        self.assertEqual(a.shape, (3, 0))

    def test_string(self):
        a = nd.array('abc', type=ndt.string)
        self.assertEqual(nd.type_of(a), ndt.string)
        a = nd.array('abc', dtype=ndt.string)
        self.assertEqual(nd.type_of(a), ndt.string)

    def test_unicode(self):
        a = nd.array(u'abc', type=ndt.string)
        self.assertEqual(nd.type_of(a), ndt.string)
        a = nd.array(u'abc', dtype=ndt.string)
        self.assertEqual(nd.type_of(a), ndt.string)

    def test_string_array(self):
        a = nd.array(['this', 'is', 'a', 'test'],
                        dtype=ndt.string)
        self.assertEqual(nd.type_of(a), ndt.type('N, string'))
        self.assertEqual(nd.as_py(a), ['this', 'is', 'a', 'test'])

        a = nd.array(['this', 'is', 'a', 'test'],
                        dtype='string("U16")')
        self.assertEqual(nd.type_of(a), ndt.type('N, string("U16")'))
        self.assertEqual(nd.as_py(a), ['this', 'is', 'a', 'test'])

    def test_unicode_array(self):
        a = nd.array([u'this', 'is', u'a', 'test'],
                        dtype=ndt.string)
        self.assertEqual(nd.type_of(a), ndt.type('N, string'))
        self.assertEqual(nd.as_py(a), ['this', 'is', 'a', 'test'])

        a = nd.array([u'this', 'is', u'a', 'test'],
                        dtype='string("U16")')
        self.assertEqual(nd.type_of(a), ndt.type('N, string("U16")'))
        self.assertEqual(nd.as_py(a), ['this', 'is', 'a', 'test'])

    def test_fixedstring_array(self):
        a = nd.array(['a', 'b', 'c'],
                        dtype='string(1,"A")')
        self.assertEqual(nd.type_of(a[0]).type_id, 'fixedstring')
        self.assertEqual(nd.type_of(a[0]).data_size, 1)
        self.assertEqual(nd.as_py(a), ['a', 'b', 'c'])

class TestStructConstruct(unittest.TestCase):
    def test_single_struct(self):
        a = nd.array([12, 'test', True], type='{x:int32; y:string; z:bool}')
        self.assertEqual(nd.type_of(a), ndt.type('{x:int32; y:string; z:bool}'))
        self.assertEqual(nd.as_py(a[0]), 12)
        self.assertEqual(nd.as_py(a[1]), 'test')
        self.assertEqual(nd.as_py(a[2]), True)

        a = nd.array({'x':12, 'y':'test', 'z':True}, type='{x:int32; y:string; z:bool}')
        self.assertEqual(nd.type_of(a), ndt.type('{x:int32; y:string; z:bool}'))
        self.assertEqual(nd.as_py(a[0]), 12)
        self.assertEqual(nd.as_py(a[1]), 'test')
        self.assertEqual(nd.as_py(a[2]), True)

    def test_nested_struct(self):
        a = nd.array([[1,2], ['test', 3.5], [3j]],
                    type='{x: 2, int16; y: {a: string; b: float64}; z: 1, cfloat32}')
        self.assertEqual(nd.as_py(a.x), [1, 2])
        self.assertEqual(nd.as_py(a.y.a), 'test')
        self.assertEqual(nd.as_py(a.y.b), 3.5)
        self.assertEqual(nd.as_py(a.z), [3j])

        a = nd.array({'x':[1,2], 'y':{'a':'test', 'b':3.5}, 'z':[3j]},
                    type='{x: 2, int16; y: {a: string; b: float64}; z: 1, cfloat32}')
        self.assertEqual(nd.as_py(a.x), [1, 2])
        self.assertEqual(nd.as_py(a.y.a), 'test')
        self.assertEqual(nd.as_py(a.y.b), 3.5)
        self.assertEqual(nd.as_py(a.z), [3j])

    def test_single_struct_array(self):
        a = nd.array([(0,0), (3,5), (12,10)], dtype='{x:int32; y:int32}')
        self.assertEqual(nd.type_of(a), ndt.type('N, {x:int32; y:int32}'))
        self.assertEqual(nd.as_py(a.x), [0, 3, 12])
        self.assertEqual(nd.as_py(a.y), [0, 5, 10])

        a = nd.array([{'x':0,'y':0}, {'x':3,'y':5}, {'x':12,'y':10}],
                    dtype='{x:int32; y:int32}')
        self.assertEqual(nd.type_of(a), ndt.type('N, {x:int32; y:int32}'))
        self.assertEqual(nd.as_py(a.x), [0, 3, 12])
        self.assertEqual(nd.as_py(a.y), [0, 5, 10])

        a = nd.array([[(3, 'X')], [(10, 'L'), (12, 'M')]],
                        dtype='{count:int32; size:string(1,"A")}')
        self.assertEqual(nd.type_of(a), ndt.type('N, var, {count:int32; size:string(1,"A")}'))
        self.assertEqual(nd.as_py(a.count), [[3], [10, 12]])
        self.assertEqual(nd.as_py(a.size), [['X'], ['L', 'M']])

        a = nd.array([[{'count':3, 'size':'X'}],
                        [{'count':10, 'size':'L'}, {'count':12, 'size':'M'}]],
                        dtype='{count:int32; size:string(1,"A")}')
        self.assertEqual(nd.type_of(a), ndt.type('N, var, {count:int32; size:string(1,"A")}'))
        self.assertEqual(nd.as_py(a.count), [[3], [10, 12]])
        self.assertEqual(nd.as_py(a.size), [['X'], ['L', 'M']])

    def test_nested_struct_array(self):
        a = nd.array([((0,1),0), ((2,2),5), ((100,10),10)],
                    dtype='{x:{a:int16; b:int16}; y:int32}')
        self.assertEqual(nd.type_of(a), ndt.type('N, {x:{a:int16; b:int16}; y:int32}'))
        self.assertEqual(nd.as_py(a.x.a), [0, 2, 100])
        self.assertEqual(nd.as_py(a.x.b), [1, 2, 10])
        self.assertEqual(nd.as_py(a.y), [0, 5, 10])

        a = nd.array([{'x':{'a':0,'b':1},'y':0},
                        {'x':{'a':2,'b':2},'y':5},
                        {'x':{'a':100,'b':10},'y':10}],
                    dtype='{x:{a:int16; b:int16}; y:int32}')
        self.assertEqual(nd.type_of(a), ndt.type('N, {x:{a:int16; b:int16}; y:int32}'))
        self.assertEqual(nd.as_py(a.x.a), [0, 2, 100])
        self.assertEqual(nd.as_py(a.x.b), [1, 2, 10])
        self.assertEqual(nd.as_py(a.y), [0, 5, 10])

        a = nd.array([[(3, ('X', 10))], [(10, ('L', 7)), (12, ('M', 5))]],
                        dtype='{count:int32; size:{name:string(1,"A"); id: int8}}')
        self.assertEqual(nd.type_of(a),
                    ndt.type('N, var, {count:int32; size:{name:string(1,"A"); id: int8}}'))
        self.assertEqual(nd.as_py(a.count), [[3], [10, 12]])
        self.assertEqual(nd.as_py(a.size.name), [['X'], ['L', 'M']])
        self.assertEqual(nd.as_py(a.size.id), [[10], [7, 5]])

    def test_missing_field(self):
        self.assertRaises(RuntimeError, nd.array,
                        [0, 1], dtype='{x:int32; y:int32; z:int32}')
        self.assertRaises(RuntimeError, nd.array,
                        {'x':0, 'z':1}, dtype='{x:int32; y:int32; z:int32}')

    def test_extra_field(self):
        self.assertRaises(RuntimeError, nd.array,
                        [0, 1, 2, 3], dtype='{x:int32; y:int32; z:int32}')
        self.assertRaises(RuntimeError, nd.array,
                        {'x':0,'y':1,'z':2,'w':3}, dtype='{x:int32; y:int32; z:int32}')

class TestIteratorConstruct(unittest.TestCase):
    # Test dynd construction from iterators
    # NumPy's np.fromiter(x, dtype) becomes nd.array(x, type='var, <dtype>')

    def test_without_specified_dtype(self):
        # Constructing from an iterator with no specified dtype
        # defaults to float64
        a = nd.array(2*x + 1 for x in range(10))
        self.assertEqual(nd.type_of(a), ndt.type('var, float64'))
        self.assertEqual(nd.as_py(a), [2*x + 1 for x in range(10)])

    def test_simple_fromiter(self):
        # Var dimension construction from a generator
        a = nd.array((2*x + 5 for x in range(10)), type='var, int32')
        self.assertEqual(nd.type_of(a), ndt.type('var, int32'))
        self.assertEqual(len(a), 10)
        self.assertEqual(nd.as_py(a), [2*x + 5 for x in range(10)])
        # Fixed dimension construction from a generator
        a = nd.array((2*x + 5 for x in range(10)), type='10, int32')
        self.assertEqual(nd.type_of(a), ndt.type('10, int32'))
        self.assertEqual(len(a), 10)
        self.assertEqual(nd.as_py(a), [2*x + 5 for x in range(10)])
        # Produce an error if it's a fixed dimension with too few elements
        self.assertRaises(nd.BroadcastError, nd.array,
                        (2*x + 5 for x in range(10)), type='11, int32')
        # Produce an error if it's a fixed dimension with too many elements
        self.assertRaises(nd.BroadcastError, nd.array,
                        (2*x + 5 for x in range(10)), type='9, int32')
        # Produce an error if it's a strided dimension
        self.assertRaises(RuntimeError, nd.array,
                        (2*x + 5 for x in range(10)), type='M, int32')

    def test_simple_fromiter_medsize(self):
        # A bigger input to exercise the dynamic resizing a bit
        a = nd.array((2*x + 5 for x in range(100000)), type='var, int32')
        self.assertEqual(nd.type_of(a), ndt.type('var, int32'))
        self.assertEqual(len(a), 100000)
        self.assertEqual(nd.as_py(a), [2*x + 5 for x in range(100000)])

    def test_ragged_fromiter(self):
        # Strided array of var from list of iterators
        a = nd.array([(1+x for x in range(3)), (5*x - 10 for x in range(5)),
                        [2, 10]], type='M, var, int32')
        self.assertEqual(nd.type_of(a), ndt.type('M, var, int32'))
        self.assertEqual(nd.as_py(a),
                        [[1,2,3], [-10, -5, 0, 5, 10], [2, 10]])
        # Var array of var from iterator of iterators
        a = nd.array(((2*x for x in range(y)) for y in range(4)),
                        type='var, var, int32')
        self.assertEqual(nd.type_of(a), ndt.type('var, var, int32'))
        self.assertEqual(nd.as_py(a), [[], [0], [0, 2], [0, 2, 4]])

    def test_uniform_fromiter(self):
        # Specify uniform type instead of full type
        a = nd.array((2*x + 1 for x in range(7)), dtype=ndt.int32)
        self.assertEqual(nd.type_of(a), ndt.type('var, int32'))
        self.assertEqual(nd.as_py(a), [2*x + 1 for x in range(7)])

class TestConstructErrors(unittest.TestCase):
    def test_bad_shape(self):
        # Too many dimensions should raise an error
        self.assertRaises(RuntimeError, nd.empty, (2,), ndt.int32)
        self.assertRaises(RuntimeError, nd.empty, (2,3), 'var, int64')

if __name__ == '__main__':
    unittest.main()

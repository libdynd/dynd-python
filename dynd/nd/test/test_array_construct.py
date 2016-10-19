import sys
import unittest
from datetime import date
from dynd import nd, ndt

class TestScalarConstructor(unittest.TestCase):
    def test_access_array_with_type(self):
        a = nd.array(1, type=ndt.int32)
        self.assertEqual(a.access_flags, 'readwrite')
#        a = nd.array(1, type=ndt.int32, access='rw')
#        self.assertEqual(a.access_flags, 'readwrite')
#        a = nd.array(1, type=ndt.int32, access='r')
#        self.assertEqual(a.access_flags, 'immutable')

    def test_access_zeros(self):
        a = nd.zeros(ndt.int32)
        self.assertEqual(a.access_flags, 'readwrite')

    def test_access_ones(self):
        a = nd.ones(ndt.int32)
        self.assertEqual(a.access_flags, 'readwrite')

class TestArrayConstruct(unittest.TestCase):
    def test_empty_array(self):
        # Empty arrays default to int32
        a = nd.array([])
        self.assertEqual(nd.type_of(a), ndt.type('0 * int32'))
        self.assertEqual(a.shape, (0,))
        a = nd.array([[], [], []])
        self.assertEqual(nd.type_of(a), ndt.type('3 * 0 * int32'))
        self.assertEqual(a.shape, (3, 0))

    def test_empty_array_dtype(self):
        a = nd.array([], type=ndt.make_fixed_dim(0, ndt.int64))
        self.assertEqual(nd.type_of(a), ndt.type('0 * int64'))
        self.assertEqual(a.shape, (0,))
#        Todo: Need to reenable this failing test
#        a = nd.array([], dtype='Fixed * float64')
#        self.assertEqual(nd.type_of(a), ndt.type('0 * float64'))
#        self.assertEqual(a.shape, (0,))
        a = nd.array([], type='var * int16')
        self.assertEqual(nd.type_of(a), ndt.type('var * int16'))
        self.assertEqual(len(a), 0)
        a = nd.array([], type='0 * int16')
        self.assertEqual(nd.type_of(a), ndt.type('0 * int16'))
        self.assertEqual(len(a), 0)
        a = nd.array([], type='0 * 3 * int16')
        self.assertEqual(nd.type_of(a), ndt.type('0 * 3 * int16'))
        self.assertEqual(a.shape, (0, 3))

class TestTypedArrayConstructors(unittest.TestCase):
    def test_empty(self):
        # Constructor from scalar type
        a = nd.empty(ndt.int32)
        self.assertEqual(a.access_flags, 'readwrite')
        self.assertEqual(nd.type_of(a), ndt.int32)
        # Constructor from type with fixed dimension
        a = nd.empty('3 * int32')
        self.assertEqual(a.access_flags, 'readwrite')
        self.assertEqual(nd.type_of(a), ndt.make_fixed_dim(3, ndt.int32))
        self.assertEqual(a.shape, (3,))
        # Constructor from type with fixed dimension, accesskwarg
        a = nd.empty('3 * int32', access='rw')
        self.assertEqual(a.access_flags, 'readwrite')
        self.assertEqual(nd.type_of(a), ndt.make_fixed_dim(3, ndt.int32))
        self.assertEqual(a.shape, (3,))
        # Constructor from shape as single integer
        a = nd.empty(3, ndt.int32)
        self.assertEqual(a.access_flags, 'readwrite')
        self.assertEqual(nd.type_of(a), ndt.type('3 * int32'))
        self.assertEqual(a.shape, (3,))
        # Constructor from shape as tuple
        a = nd.empty((3,4), ndt.int32)
        self.assertEqual(a.access_flags, 'readwrite')
        self.assertEqual(nd.type_of(a), ndt.type('3 * 4 * int32'))
        self.assertEqual(a.shape, (3,4))
        # Constructor from shape as variadic arguments
        a = nd.empty(3, 4, ndt.int32)
        self.assertEqual(a.access_flags, 'readwrite')
        self.assertEqual(nd.type_of(a), ndt.type('3 * 4 * int32'))
        self.assertEqual(a.shape, (3,4))
        # Constructor from shape as variadic arguments, access kwarg
        a = nd.empty(3, 4, ndt.int32, access='rw')
        self.assertEqual(a.access_flags, 'readwrite')
        self.assertEqual(nd.type_of(a), ndt.type('3 * 4 * int32'))
        self.assertEqual(a.shape, (3,4))

    def check_constructor(self, cons, value):
        # Constructor from scalar type
        a = cons(ndt.int32)
        self.assertEqual(a.access_flags, 'readwrite')
        self.assertEqual(nd.type_of(a), ndt.int32)
        self.assertEqual(nd.as_py(a), value)
        # Constructor from type with fixed dimension
        a = cons('3 * int32')
        self.assertEqual(a.access_flags, 'readwrite')
        self.assertEqual(nd.type_of(a), ndt.make_fixed_dim(3, ndt.int32))
        self.assertEqual(a.shape, (3,))
        self.assertEqual(nd.as_py(a), [value]*3)
        # Constructor from shape as single integer
        a = cons(3, ndt.int32)
        self.assertEqual(a.access_flags, 'readwrite')
        self.assertEqual(nd.type_of(a), ndt.type('3 * int32'))
        self.assertEqual(a.shape, (3,))
        self.assertEqual(nd.as_py(a), [value]*3)
        # Constructor from shape as tuple
        a = cons((3,4), ndt.int32)
        self.assertEqual(a.access_flags, 'readwrite')
        self.assertEqual(nd.type_of(a), ndt.type('3 * 4 * int32'))
        self.assertEqual(a.shape, (3,4))
        self.assertEqual(nd.as_py(a), [[value]*4]*3)
        # Constructor from shape as variadic arguments
        a = cons(3, 4, ndt.int32)
        self.assertEqual(a.access_flags, 'readwrite')
        self.assertEqual(nd.type_of(a), ndt.type('3 * 4 * int32'))
        self.assertEqual(a.shape, (3,4))
        self.assertEqual(nd.as_py(a), [[value]*4]*3)
        # Constructor of a struct type

    def test_zeros(self):
        self.check_constructor(nd.zeros, 0)

    def test_ones(self):
        self.check_constructor(nd.ones, 1)

class TestArrayConstructor(unittest.TestCase):
    # Always constructs a new array
    def test_simple(self):
        a = nd.array([1, 2, 3])
        self.assertEqual(nd.type_of(a), ndt.type('3 * int32'))
        # Modifying 'a' shouldn't affect 'b', because it's a copy
        b = nd.array(a)
        a[1] = 10
        self.assertEqual(nd.as_py(b), [1, 2, 3])

    def test_access_from_pyobject(self):
        a = nd.array([1, 2, 3])
        self.assertEqual(a.access_flags, 'readwrite')
#        a = nd.array([1, 2, 3], access='immutable')
#        self.assertEqual(a.access_flags, 'immutable')
#        a = nd.array([1, 2, 3], access='readonly')
#        self.assertEqual(a.access_flags, 'immutable')
#        a = nd.array([1, 2, 3], access='r')
#        self.assertEqual(a.access_flags, 'immutable')
#        a = nd.array([1, 2, 3], access='readwrite')
#        self.assertEqual(a.access_flags, 'readwrite')
#        a = nd.array([1, 2, 3], access='rw')
#        self.assertEqual(a.access_flags, 'readwrite')

    """
    def test_access_from_immutable_array(self):
        # `a` is an immutable array
        a = nd.array([1, 2, 3], access='r')
        self.assertEqual(a.access_flags, 'immutable')
        b = nd.array(a)
        self.assertEqual(b.access_flags, 'readwrite')
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
    """

    def test_access_from_readwrite_array(self):
        # `a` is a readwrite array
        a = nd.array([1, 2, 3])
        self.assertEqual(a.access_flags, 'readwrite')
        b = nd.array(a)
        self.assertEqual(b.access_flags, 'readwrite')
    #    b = nd.array(a, access='immutable')
    #    self.assertEqual(b.access_flags, 'immutable')
    #    b = nd.array(a, access='readonly')
    #    self.assertEqual(b.access_flags, 'immutable')
    #    b = nd.array(a, access='r')
    #    self.assertEqual(b.access_flags, 'immutable')
    #    b = nd.array(a, access='readwrite')
    #    self.assertEqual(b.access_flags, 'readwrite')
    #    b = nd.array(a, access='rw')
    #    self.assertEqual(b.access_flags, 'readwrite')

class TestViewConstructor(unittest.TestCase):
    # Always constructs a view
    def test_simple(self):
        a = nd.array([1, 2, 3])
        # Modifying 'a' should affect 'b', because it's a view
        b = nd.view(a)
        a[1] = 10
        self.assertEqual(nd.as_py(b), [1, 10, 3])
        # Can't construct a view of a python list
        self.assertRaises(TypeError, nd.view, [1, 2, 3])

class TestAsArrayConstructor(unittest.TestCase):
    # Constructs a view if possible, otherwise a copy
    def test_simple(self):
        a = nd.asarray([1, 2, 3])
        self.assertEqual(nd.type_of(a), ndt.type('3 * int32'))

        # Modifying 'a' should affect 'b', because it's a view
        b = nd.asarray(a)
        self.assertEqual(nd.as_py(b), [1, 2, 3])
        a[1] = 10
        self.assertEqual(nd.as_py(b), [1, 10, 3])

        # asarray no longer supports changing the access flags.
        # Once a different api is available in the Python bindings
        # for changing the access flags, these tests should be rewritten.
        """# Can take a readonly view, but still modify the original
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
        self.assertEqual(nd.as_py(b), [1, 40, 3])"""

class TestStringConstruct(unittest.TestCase):
    def test_string(self):
        a = nd.array('abc', type=ndt.string)
        self.assertEqual(nd.type_of(a), ndt.string)

    def test_unicode(self):
        a = nd.array(u'abc', type=ndt.string)
        self.assertEqual(nd.type_of(a), ndt.string)

    def test_string_array(self):
        a = nd.array(['this', 'is', 'a', 'test'],
                        type=ndt.make_fixed_dim(4, ndt.string))
        self.assertEqual(nd.type_of(a), ndt.type('4 * string'))
        self.assertEqual(nd.as_py(a), ['this', 'is', 'a', 'test'])

    def test_unicode_array(self):
        a = nd.array([u'this', 'is', u'a', 'test'],
                        type=ndt.make_fixed_dim(4, ndt.string))
        self.assertEqual(nd.type_of(a), ndt.type('4 * string'))
        self.assertEqual(nd.as_py(a), ['this', 'is', 'a', 'test'])

    def test_fixed_string_array(self):
        a = nd.array(['a', 'b', 'c'],
                        type='3 * fixed_string[1,"A"]')
        self.assertEqual(nd.type_of(a[0]).data_size, 1)
        self.assertEqual(nd.as_py(a), ['a', 'b', 'c'])

class TestStructConstruct(unittest.TestCase):
    def test_single_struct(self):
        a = nd.array([12, 'test', True], type='{x:int32, y:string, z:bool}')
        self.assertEqual(nd.type_of(a), ndt.type('{x:int32, y:string, z:bool}'))
        self.assertEqual(nd.as_py(a[0]), 12)
        self.assertEqual(nd.as_py(a[1]), 'test')
        self.assertEqual(nd.as_py(a[2]), True)

        a = nd.array({'x':12, 'y':'test', 'z':True}, type='{x:int32, y:string, z:bool}')
        self.assertEqual(nd.type_of(a), ndt.type('{x:int32, y:string, z:bool}'))
        self.assertEqual(nd.as_py(a[0]), 12)
        self.assertEqual(nd.as_py(a[1]), 'test')
        self.assertEqual(nd.as_py(a[2]), True)

    def test_nested_struct(self):
        a = nd.array([[1,2], ['test', 3.5], [3j]],
                    type='{x: 2 * int16, y: {a: string, b: float64}, z: 1 * complex[float32]}')
        self.assertEqual(nd.as_py(a.x), [1, 2])
        self.assertEqual(nd.as_py(a.y.a), 'test')
        self.assertEqual(nd.as_py(a.y.b), 3.5)
        self.assertEqual(nd.as_py(a.z), [3j])

        a = nd.array({'x':[1,2], 'y':{'a':'test', 'b':3.5}, 'z':[3j]},
                    type='{x: 2 * int16, y: {a: string, b: float64}, z: 1 * complex[float64]}')
        self.assertEqual(nd.as_py(a.x), [1, 2])
        self.assertEqual(nd.as_py(a.y.a), 'test')
        self.assertEqual(nd.as_py(a.y.b), 3.5)
        self.assertEqual(nd.as_py(a.z), [3j])

    def test_single_tuple_array(self):
        a = nd.array([(0,0), (3,5), (12,10)], type='3 * (int32, int32)')
        self.assertEqual(nd.type_of(a), ndt.type('3 * (int32, int32)'))
        self.assertEqual(nd.as_py(a[:,0]), [0, 3, 12])
        self.assertEqual(nd.as_py(a[:,1]), [0, 5, 10])

    def test_single_struct_array(self):
        a = nd.array([(0,0), (3,5), (12,10)], type='3 * {x:int32, y:int32}')
        self.assertEqual(nd.type_of(a), ndt.type('3 * {x:int32, y:int32}'))
        self.assertEqual(nd.as_py(a.x), [0, 3, 12])
        self.assertEqual(nd.as_py(a.y), [0, 5, 10])

        a = nd.array([{'x':0,'y':0}, {'x':3,'y':5}, {'x':12,'y':10}],
                    type='3 * {x:int32, y:int32}')
        self.assertEqual(nd.type_of(a), ndt.type('3 * {x:int32, y:int32}'))
        self.assertEqual(nd.as_py(a.x), [0, 3, 12])
        self.assertEqual(nd.as_py(a.y), [0, 5, 10])

        a = nd.array([[(3, 'X')], [(10, 'L'), (12, 'M')]],
                        type='2 * var * {count:int32, size:fixed_string[1,"A"]}')
        self.assertEqual(nd.type_of(a),
                       ndt.type('2 * var * {count:int32, size:fixed_string[1,"A"]}'))
        self.assertEqual(nd.as_py(a.count), [[3], [10, 12]])
        self.assertEqual(nd.as_py(a.size), [['X'], ['L', 'M']])

        a = nd.array([[{'count':3, 'size':'X'}],
                        [{'count':10, 'size':'L'}, {'count':12, 'size':'M'}]],
                        type='2 * var * {count:int32, size:fixed_string[1,"A"]}')
        self.assertEqual(nd.type_of(a), ndt.type('2 * var * {count:int32, size:fixed_string[1,"A"]}'))
        self.assertEqual(nd.as_py(a.count), [[3], [10, 12]])
        self.assertEqual(nd.as_py(a.size), [['X'], ['L', 'M']])

    def test_nested_struct_array(self):
        a = nd.array([((0,1),0), ((2,2),5), ((100,10),10)],
                    type='3 * {x:{a:int16, b:int16}, y:int32}')
        self.assertEqual(nd.type_of(a), ndt.type('3 * {x:{a:int16, b:int16}, y:int32}'))
        self.assertEqual(nd.as_py(a.x.a), [0, 2, 100])
        self.assertEqual(nd.as_py(a.x.b), [1, 2, 10])
        self.assertEqual(nd.as_py(a.y), [0, 5, 10])

        a = nd.array([{'x':{'a':0,'b':1},'y':0},
                        {'x':{'a':2,'b':2},'y':5},
                        {'x':{'a':100,'b':10},'y':10}],
                    type='3 * {x:{a:int16, b:int16}, y:int32}')
        self.assertEqual(nd.type_of(a), ndt.type('3 * {x:{a:int16, b:int16}, y:int32}'))
        self.assertEqual(nd.as_py(a.x.a), [0, 2, 100])
        self.assertEqual(nd.as_py(a.x.b), [1, 2, 10])
        self.assertEqual(nd.as_py(a.y), [0, 5, 10])

        a = nd.array([[(3, ('X', 10))], [(10, ('L', 7)), (12, ('M', 5))]],
                        type='2 * var * {count:int32, size:{name:fixed_string[1,"A"], id: int8}}')
        self.assertEqual(nd.type_of(a),
                    ndt.type('2 * var * {count:int32, size:{name:fixed_string[1,"A"], id: int8}}'))
        self.assertEqual(nd.as_py(a.count), [[3], [10, 12]])
        self.assertEqual(nd.as_py(a.size.name), [['X'], ['L', 'M']])
        self.assertEqual(nd.as_py(a.size.id), [[10], [7, 5]])

    def test_missing_field(self):
        self.assertRaises(nd.BroadcastError, nd.array,
                        [0, 1], type='{x:int32, y:int32, z:int32}')
        self.assertRaises(nd.BroadcastError, nd.array,
                        {'x':0, 'z':1}, type='{x:int32, y:int32, z:int32}')

    def test_extra_field(self):
        self.assertRaises(nd.BroadcastError, nd.array,
                        [0, 1, 2, 3], type='{x:int32, y:int32, z:int32}')
        self.assertRaises(nd.BroadcastError, nd.array,
                        {'x':0,'y':1,'z':2,'w':3}, type='{x:int32, y:int32, z:int32}')

#class TestIteratorConstruct(unittest.TestCase):
#    # Test dynd construction from iterators
#    #
#    # NumPy's np.fromiter(x, dtype) becomes
#    #         nd.array(x, type=ndt.make_var(dtype)')
#    #
#    # Additionally, dynd supports dynamically deducing the type as
#    # it processes the iterators, so nd.array(x) where x is an iterator
#    # should work well too.
#
#    def test_dynamic_fromiter_notype(self):
#        # When constructing from an empty iterator, defaults to int32
#        a = nd.array(x for x in [])
#        self.assertEqual(nd.type_of(a), ndt.type('0 * int32'))
#        self.assertEqual(nd.as_py(a), [])
#
#    def test_dynamic_fromiter_onetype(self):
#        # Constructing with an iterator like this uses a dynamic
#        # array construction method. In this simple case, we
#        # use generators that have a consistent type
#        # bool result
#        a = nd.array(iter([True, False]))
#        self.assertEqual(nd.type_of(a), ndt.type('2 * bool'))
#        self.assertEqual(nd.as_py(a), [True, False])
#        # int32 result
#        a = nd.array(iter([1, 2, True, False]))
#        self.assertEqual(nd.type_of(a), ndt.type('4 * int32'))
#        self.assertEqual(nd.as_py(a), [1, 2, 1, 0])
#        # int64 result
#        a = nd.array(iter([10000000000, 1, 2, True, False]))
#        self.assertEqual(nd.type_of(a), ndt.type('5 * int64'))
#        self.assertEqual(nd.as_py(a), [10000000000, 1, 2, 1, 0])
#        # float64 result
#        a = nd.array(iter([3.25, 10000000000, 1, 2, True, False]))
#        self.assertEqual(nd.type_of(a), ndt.type('6 * float64'))
#        self.assertEqual(nd.as_py(a), [3.25, 10000000000, 1, 2, 1, 0])
#        # complex[float64] result
#        a = nd.array(iter([3.25j, 3.25, 10000000000, 1, 2, True, False]))
#        self.assertEqual(nd.type_of(a), ndt.type('7 * complex[float64]'))
#        self.assertEqual(nd.as_py(a), [3.25j, 3.25, 10000000000, 1, 2, 1, 0])
#
#        """
#        Todo: Reenable this with new strings
#
#        # string result
#        a = nd.array(str(x) + 'test' for x in range(10))
#        self.assertEqual(nd.type_of(a), ndt.type('10 * string'))
#        self.assertEqual(nd.as_py(a), [str(x) + 'test' for x in range(10)])
#        # string result
#        a = nd.array(iter([u'test', 'test2']))
#        self.assertEqual(nd.type_of(a), ndt.type('2 * string'))
#        self.assertEqual(nd.as_py(a), [u'test', u'test2'])
#        # bytes result
#        if sys.version_info[0] >= 3:
#            a = nd.array(b'x'*x for x in range(10))
#            self.assertEqual(nd.type_of(a), ndt.type('10 * bytes'))
#            self.assertEqual(nd.as_py(a), [b'x'*x for x in range(10)])
#        """
#
#    def test_dynamic_fromiter_booltypepromo(self):
#        # Test iterator construction cases promoting from a boolean
#        # int32 result
#        a = nd.array(iter([True, False, 3]))
#        self.assertEqual(nd.type_of(a), ndt.type('3 * int32'))
#        self.assertEqual(nd.as_py(a), [1, 0, 3])
#        # int64 result
#        a = nd.array(iter([True, False, -10000000000]))
#        self.assertEqual(nd.type_of(a), ndt.type('3 * int64'))
#        self.assertEqual(nd.as_py(a), [1, 0, -10000000000])
#        # float64 result
#        a = nd.array(iter([True, False, 3.25]))
#        self.assertEqual(nd.type_of(a), ndt.type('3 * float64'))
#        self.assertEqual(nd.as_py(a), [1, 0, 3.25])
#        # complex[float64] result
#        a = nd.array(iter([True, False, 3.25j]))
#        self.assertEqual(nd.type_of(a), ndt.type('3 * complex[float64]'))
#        self.assertEqual(nd.as_py(a), [1, 0, 3.25j])
#        # Should raise an error mixing bool and string/bytes
#        self.assertRaises(TypeError, nd.array, iter([True, False, "test"]))
#        self.assertRaises(TypeError, nd.array, iter([True, False, u"test"]))
#        self.assertRaises(TypeError, nd.array, iter([True, False, b"test"]))
#
#    def test_dynamic_fromiter_int32typepromo(self):
#        # Test iterator construction cases promoting from an int32
#        # int64 result
#        a = nd.array(iter([1, 2, 10000000000]))
#        self.assertEqual(nd.type_of(a), ndt.type('3 * int64'))
#        self.assertEqual(nd.as_py(a), [1, 2, 10000000000])
#        # float64 result
#        a = nd.array(iter([1, 2, 3.25]))
#        self.assertEqual(nd.type_of(a), ndt.type('3 * float64'))
#        self.assertEqual(nd.as_py(a), [1, 2, 3.25])
#        # complex[float64] result
#        a = nd.array(iter([1, 2, 3.25j]))
#        self.assertEqual(nd.type_of(a), ndt.type('3 * complex[float64]'))
#        self.assertEqual(nd.as_py(a), [1, 2, 3.25j])
#        # Should raise an error mixing int32 and string/bytes
#        self.assertRaises(TypeError, nd.array, iter([1, 2, "test"]))
#        self.assertRaises(TypeError, nd.array, iter([1, 2, u"test"]))
#        self.assertRaises(TypeError, nd.array, iter([1, 2, b"test"]))
#
#    def test_dynamic_fromiter_int64typepromo(self):
#        # Test iterator construction cases promoting from an int64
#        # float64 result
#        a = nd.array(iter([10000000000, 2, 3.25]))
#        self.assertEqual(nd.type_of(a), ndt.type('3 * float64'))
#        self.assertEqual(nd.as_py(a), [10000000000, 2, 3.25])
#        # complex[float64] result
#        a = nd.array(iter([10000000000, 2, 3.25j]))
#        self.assertEqual(nd.type_of(a), ndt.type('3 * complex[float64]'))
#        self.assertEqual(nd.as_py(a), [10000000000, 2, 3.25j])
#        # Should raise an error mixing int64 and string/bytes
#        self.assertRaises(TypeError, nd.array, iter([10000000000, 2, "test"]))
#        self.assertRaises(TypeError, nd.array, iter([10000000000, 2, u"test"]))
#        self.assertRaises(TypeError, nd.array, iter([10000000000, 2, b"test"]))
#
#    def test_dynamic_fromiter_float64typepromo(self):
#        # Test iterator construction cases promoting from an float64
#        # complex[float64] result
#        a = nd.array(iter([3.25, 2, 3.25j]))
#        self.assertEqual(nd.type_of(a), ndt.type('3 * complex[float64]'))
#        self.assertEqual(nd.as_py(a), [3.25, 2, 3.25j])
#        # Should raise an error mixing float64 and string/bytes
#        self.assertRaises(TypeError, nd.array, iter([3.25, 2, "test"]))
#        self.assertRaises(TypeError, nd.array, iter([3.25, 2, u"test"]))
#        self.assertRaises(TypeError, nd.array, iter([3.25, 2, b"test"]))
#
#    def test_dynamic_fromiter_complexfloat64typepromo(self):
#        # Test iterator construction cases promoting from an complex[float64]
#        # Should raise an error mixing complex[float64] and string/bytes
#        self.assertRaises(TypeError, nd.array, iter([3.25j, 2, "test"]))
#        self.assertRaises(TypeError, nd.array, iter([3.25j, 2, u"test"]))
#        self.assertRaises(TypeError, nd.array, iter([3.25j, 2, b"test"]))
#
#    def test_simple_fromiter(self):
#        # Var dimension construction from a generator
#        a = nd.array((2*x + 5 for x in range(10)), type='var * int32')
#        self.assertEqual(nd.type_of(a), ndt.type('var * int32'))
#        self.assertEqual(len(a), 10)
#        self.assertEqual(nd.as_py(a), [2*x + 5 for x in range(10)])
#        # Fixed dimension construction from a generator
#        a = nd.array((2*x + 5 for x in range(10)), type='10 * int32')
#        self.assertEqual(nd.type_of(a), ndt.type('10 * int32'))
#        self.assertEqual(len(a), 10)
#        self.assertEqual(nd.as_py(a), [2*x + 5 for x in range(10)])
#        # Produce an error if it's a fixed dimension with too few elements
#        self.assertRaises(nd.BroadcastError, nd.array,
#                        (2*x + 5 for x in range(10)), type='11 * int32')
#        # Produce an error if it's a fixed dimension with too many elements
#        self.assertRaises(nd.BroadcastError, nd.array,
#                        (2*x + 5 for x in range(10)), type='9 * int32')
#        # Produce an error if it's a fixed dimension
#        self.assertRaises(TypeError, nd.array,
#                        (2*x + 5 for x in range(10)), type='Fixed * int32')
#
#    def test_simple_fromiter_medsize(self):
#        # A bigger input to exercise the dynamic resizing a bit
#        a = nd.array((2*x + 5 for x in range(100000)), type='var * int32')
#        self.assertEqual(nd.type_of(a), ndt.type('var * int32'))
#        self.assertEqual(len(a), 100000)
#        self.assertEqual(nd.as_py(a), [2*x + 5 for x in range(100000)])
#
#    def test_ragged_fromiter(self):
#        # Strided array of var from list of iterators
#        a = nd.array([(1+x for x in range(3)), (5*x - 10 for x in range(5)),
#                        [2, 10]], type='3 * var * int32')
#        self.assertEqual(nd.type_of(a), ndt.type('3 * var * int32'))
#        self.assertEqual(nd.as_py(a),
#                        [[1,2,3], [-10, -5, 0, 5, 10], [2, 10]])
#        # Var array of var from iterator of iterators
#        a = nd.array(((2*x for x in range(y)) for y in range(4)),
#                        type='var * var * int32')
#        self.assertEqual(nd.type_of(a), ndt.type('var * var * int32'))
#        self.assertEqual(nd.as_py(a), [[], [0], [0, 2], [0, 2, 4]])
#        # Range of ranges
#        a = nd.array(range(i) for i in range(4))
#        self.assertEqual(nd.as_py(a), [list(range(i)) for i in range(4)])
#
#    def test_ragged_fromiter_typepromo(self):
#        # 2D nested iterators
#        vals = [[True, False],
#                [False, 2, 3],
#                [-10000000000],
#                [True, 10, 3.125, 5.5j]]
#        a = nd.array(iter(x) for x in vals)
#        self.assertEqual(nd.type_of(a), ndt.type('4 * var * complex[float64]'))
#        self.assertEqual(nd.as_py(a), vals)
#        # 3D nested iterators
#        vals = [[[True, True, True],
#                 [False, False]],
#                [[True, True, False],
#                 [False, False, -1000, 10000000000],
#                 [10, 20, 10]],
#                [],
#                [[],
#                 [1.5],
#                 []]]
#        a = nd.array((iter(y) for y in x) for x in vals)
#        self.assertEqual(nd.type_of(a), ndt.type('4 * var * var * float64'))
#        self.assertEqual(nd.as_py(a), vals)
#        # Iterator of lists
#        vals = [[True, 2, 3],
#                [4, 5, 6.5],
#                [1, 2, 3]]
#        a = nd.array(iter(vals))
#        self.assertEqual(nd.type_of(a), ndt.type('3 * 3 * float64'))
#        self.assertEqual(nd.as_py(a), vals)
#        # Iterator starting with list, also including iterator
#        vals = [[True, 2, 3],
#                [4, 5, 6.5],
#                [1, 2, 3]]
#        a = nd.array(x for x in [vals[0], iter(vals[1]), vals[2]])
#        self.assertEqual(nd.type_of(a), ndt.type('3 * 3 * float64'))
#        self.assertEqual(nd.as_py(a), vals)
#        # Iterator with lists, but ragged
#        vals = [[1], [2, 3, 4], [5, 6]]
#        a = nd.array(iter(vals))
#        self.assertEqual(nd.type_of(a), ndt.type('3 * var * int32'))
#        self.assertEqual(nd.as_py(a), vals)
#        # Iterator starting with list, first raggedness is a short iterator
#        vals = [[1, 2, 3], [4], [5, 6]]
#        a = nd.array(x for x in [vals[0], iter(vals[1]), vals[2]])
#        self.assertEqual(nd.type_of(a), ndt.type('3 * var * int32'))
#        self.assertEqual(nd.as_py(a), vals)
#        # Iterator starting with list, first raggedness is a long iterator
#        vals = [[1], [2, 3, 4], [5, 6]]
#        a = nd.array(x for x in [vals[0], iter(vals[1]), vals[2]])
#        self.assertEqual(nd.type_of(a), ndt.type('3 * var * int32'))
#        self.assertEqual(nd.as_py(a), vals)
#
#    def test_ragged_fromlistofiter_typepromo(self):
#        # list of iterators
#        vals = [[True, False],
#                [False, 2, 3],
#                [-10000000000],
#                [True, 10, 3.125, 5.5j]]
#        a = nd.array([iter(x) for x in vals])
#        self.assertEqual(nd.type_of(a), ndt.type('4 * var * complex[float64]'))
#        self.assertEqual(nd.as_py(a), vals)
#        # list of list/iterator
#        a = nd.array([[1,2,3], (1.5*x for x in range(4)), iter([-1, 1])])
#        self.assertEqual(nd.type_of(a), ndt.type('3 * var * float64'))
#        self.assertEqual(nd.as_py(a),
#                         [[1,2,3], [1.5*x for x in range(4)], [-1,1]])
#
#    def test_ragged_initial_empty_typepromo(self):
#        # iterator of lists, first one is empty
#        vals = [[],
#                [False, 2, 3]]
#        a = nd.array(iter(x) for x in vals)
#        self.assertEqual(nd.type_of(a), ndt.type('2 * var * int32'))
#        self.assertEqual(nd.as_py(a), vals)
#
#    def test_dtype_fromiter(self):
#        # Specify dtype instead of full type
#        a = nd.array((2*x + 1 for x in range(7)), type=ndt.make_var_dim(ndt.int32))
#        self.assertEqual(nd.type_of(a), ndt.type('var * int32'))
#        self.assertEqual(nd.as_py(a), [2*x + 1 for x in range(7)])

class TestDeduceDims(unittest.TestCase):
    """
    def test_simplearr(self):
        val = [[[1, 2], [3, 4]], [[5, 6], [7, 8]],
               [[11, 12], [13, 14]], [[15, 16], [17, 18]]]
        # Deduce all the dims
        a = nd.array(val, type=ndt.int16)
        self.assertEqual(nd.type_of(a), ndt.type('4 * 2 * 2 * int16'))
        self.assertEqual(nd.as_py(a), val)
        # Specify some dims as fixed
        a = nd.array(val, type='Fixed * int16')
        self.assertEqual(nd.type_of(a), ndt.type('4 * 2 * 2 * int16'))
        self.assertEqual(nd.as_py(a), val)
        a = nd.array(val, type='Fixed * Fixed * int16')
        self.assertEqual(nd.type_of(a), ndt.type('4 * 2 * 2 * int16'))
        self.assertEqual(nd.as_py(a), val)
        a = nd.array(val, type='Fixed * Fixed * Fixed * int16')
        self.assertEqual(nd.type_of(a), ndt.type('4 * 2 * 2 * int16'))
        self.assertEqual(nd.as_py(a), val)
        # Specify some dims as fixed
        a = nd.array(val, type='2 * int16')
        self.assertEqual(nd.type_of(a), ndt.type('4 * 2 * 2 * int16'))
        self.assertEqual(nd.as_py(a), val)
        a = nd.array(val, type='2 * 2 * int16')
        self.assertEqual(nd.type_of(a), ndt.type('4 * 2 * 2 * int16'))
        self.assertEqual(nd.as_py(a), val)
        a = nd.array(val, type='4 * 2 * 2 * int16')
        self.assertEqual(nd.type_of(a), ndt.type('4 * 2 * 2 * int16'))
        self.assertEqual(nd.as_py(a), val)
        # Mix fixed, symbolic fixed, and var
        a = nd.array(val, type='4 * var * Fixed * int16')
        self.assertEqual(nd.type_of(a), ndt.type('4 * var * 2 * int16'))
        self.assertEqual(nd.as_py(a), val)
        a = nd.array(val, type='var * 2 * int16')
        self.assertEqual(nd.type_of(a), ndt.type('4 * var * 2 * int16'))
        self.assertEqual(nd.as_py(a), val)
        a = nd.array(val, type='Fixed * 2 * int16')
        self.assertEqual(nd.type_of(a), ndt.type('4 * 2 * 2 * int16'))
        self.assertEqual(nd.as_py(a), val)
    """

    def test_empty(self):
        # A fixed dimension of zero size gets absorbed
        a = nd.array([], type='0 * int32')
        self.assertEqual(nd.type_of(a), ndt.type('0 * int32'))
        self.assertEqual(nd.as_py(a), [])
        # A symbolic fixed dimension gets absorbed
#       Todo: Need to reenable this failing test
#        a = nd.array([], dtype='Fixed * int32')
#        self.assertEqual(nd.type_of(a), ndt.type('0 * int32'))
 #       self.assertEqual(nd.as_py(a), [])
        # A var dimension gets absorbed
        a = nd.array([], type='var * int32')
        self.assertEqual(nd.type_of(a), ndt.type('var * int32'))
        self.assertEqual(nd.as_py(a), [])

class TestConstructErrors(unittest.TestCase):
    def test_bad_params(self):
        self.assertRaises(TypeError, nd.array, type='int32')
        self.assertRaises(TypeError, nd.array, type='2 * 2 * int32')

    def test_dict_auto_detect(self):
        # Trigger failure in initial auto detect pass
        self.assertRaises(ValueError, nd.array, {'x' : 1})
        self.assertRaises(ValueError, nd.array, [{'x' : 1}])
        # Trigger failure in later type promotion
        # TODO: fix
        # self.assertRaises(ValueError, nd.array, [['a'], {'x' : 1}])

class TestOptionArrayConstruct(unittest.TestCase):
    def check_scalars(self, type, input_expected):
        type = ndt.type(type)
        for input, expected in input_expected:
            a = nd.array(input, type=type)
            self.assertEqual(nd.type_of(a), type)
            self.assertEqual(nd.as_py(a), expected)

    def test_scalar_option(self):
        self.check_scalars('?bool', [(None, None),
                                     #('', None),
                                     #('NA', None),
                                     (False, False),
                                     ('true', True)])
        self.check_scalars('?int', [(None, None), ('', None), ('NA', None), (-10, -10), ('12354', 12354)])
        self.check_scalars('?real', [(None, None),
                                     ('', None),
                                     ('NA', None),
                                     (-10, -10),
                                     ('12354', 12354),
                                     (1.25, 1.25),
                                     ('125e20', 125e20)])
        #self.check_scalars('?string', [(None, None),
        #                               ('', ''),
        #                               ('NA', 'NA'),
        #                               (u'\uc548\ub155', u'\uc548\ub155')
        #                               ])

if __name__ == '__main__':
    unittest.main(verbosity=2)

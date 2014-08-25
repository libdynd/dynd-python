import sys
import unittest
from dynd import nd, ndt
import numpy as np
from datetime import date, datetime
from numpy.testing import *

class TestNumpyDTypeInterop(unittest.TestCase):
    def setUp(self):
        if sys.byteorder == 'little':
            self.nonnative = '>'
        else:
            self.nonnative = '<'

    def test_ndt_type_from_numpy_scalar_types(self):
        # Tests converting numpy scalar types to dynd types
        self.assertEqual(ndt.bool, ndt.type(np.bool))
        self.assertEqual(ndt.bool, ndt.type(np.bool_))
        self.assertEqual(ndt.int8, ndt.type(np.int8))
        self.assertEqual(ndt.int16, ndt.type(np.int16))
        self.assertEqual(ndt.int32, ndt.type(np.int32))
        self.assertEqual(ndt.int64, ndt.type(np.int64))
        self.assertEqual(ndt.uint8, ndt.type(np.uint8))
        self.assertEqual(ndt.uint16, ndt.type(np.uint16))
        self.assertEqual(ndt.uint32, ndt.type(np.uint32))
        self.assertEqual(ndt.uint64, ndt.type(np.uint64))
        self.assertEqual(ndt.float32, ndt.type(np.float32))
        self.assertEqual(ndt.float64, ndt.type(np.float64))
        self.assertEqual(ndt.complex_float32, ndt.type(np.complex64))
        self.assertEqual(ndt.complex_float64, ndt.type(np.complex128))

    def test_ndt_type_from_numpy_dtype(self):
        # Tests converting numpy dtypes to dynd types
        # native byte order
        self.assertEqual(ndt.bool, ndt.type(np.dtype(np.bool)))
        self.assertEqual(ndt.int8, ndt.type(np.dtype(np.int8)))
        self.assertEqual(ndt.int16, ndt.type(np.dtype(np.int16)))
        self.assertEqual(ndt.int32, ndt.type(np.dtype(np.int32)))
        self.assertEqual(ndt.int64, ndt.type(np.dtype(np.int64)))
        self.assertEqual(ndt.uint8, ndt.type(np.dtype(np.uint8)))
        self.assertEqual(ndt.uint16, ndt.type(np.dtype(np.uint16)))
        self.assertEqual(ndt.uint32, ndt.type(np.dtype(np.uint32)))
        self.assertEqual(ndt.uint64, ndt.type(np.dtype(np.uint64)))
        self.assertEqual(ndt.float32, ndt.type(np.dtype(np.float32)))
        self.assertEqual(ndt.float64, ndt.type(np.dtype(np.float64)))
        self.assertEqual(ndt.complex_float32, ndt.type(np.dtype(np.complex64)))
        self.assertEqual(ndt.complex_float64, ndt.type(np.dtype(np.complex128)))
        self.assertEqual(ndt.make_fixedstring(10, 'ascii'),
                    ndt.type(np.dtype('S10')))
        self.assertEqual(ndt.make_fixedstring(10, 'utf_32'),
                    ndt.type(np.dtype('U10')))

        # non-native byte order
        nonnative = self.nonnative

        self.assertEqual(ndt.make_byteswap(ndt.int16),
                ndt.type(np.dtype(nonnative + 'i2')))
        self.assertEqual(ndt.make_byteswap(ndt.int32),
                ndt.type(np.dtype(nonnative + 'i4')))
        self.assertEqual(ndt.make_byteswap(ndt.int64),
                ndt.type(np.dtype(nonnative + 'i8')))
        self.assertEqual(ndt.make_byteswap(ndt.uint16),
                ndt.type(np.dtype(nonnative + 'u2')))
        self.assertEqual(ndt.make_byteswap(ndt.uint32),
                ndt.type(np.dtype(nonnative + 'u4')))
        self.assertEqual(ndt.make_byteswap(ndt.uint64),
                ndt.type(np.dtype(nonnative + 'u8')))
        self.assertEqual(ndt.make_byteswap(ndt.float32),
                ndt.type(np.dtype(nonnative + 'f4')))
        self.assertEqual(ndt.make_byteswap(ndt.float64),
                ndt.type(np.dtype(nonnative + 'f8')))
        self.assertEqual(ndt.make_byteswap(ndt.complex_float32),
                ndt.type(np.dtype(nonnative + 'c8')))
        self.assertEqual(ndt.make_byteswap(ndt.complex_float64),
                ndt.type(np.dtype(nonnative + 'c16')))

    def test_ndt_type_from_numpy_dtype_struct(self):
        # aligned struct
        tp0 = ndt.type(np.dtype([('x', np.int32), ('y', np.int64)],
                            align=True))
        tp1 = ndt.type('c{x : int32, y : int64}')
        self.assertEqual(tp0, tp1)
        # unaligned struct
        tp0 = ndt.type(np.dtype([('x', np.int32), ('y', np.int64)]))
        tp1 = ndt.make_cstruct([ndt.make_unaligned(ndt.int32),
                        ndt.make_unaligned(ndt.int64)],
                        ['x', 'y'])
        self.assertEqual(tp0, tp1)

    def test_ndt_type_from_h5py_special(self):
        # h5py 2.3 style "special dtype"
        dt = np.dtype(object, metadata={'vlen' : str})
        self.assertEqual(ndt.type(dt), ndt.string)
        if sys.version_info < (3, 0):
            dt = np.dtype(object, metadata={'vlen' : unicode})
            self.assertEqual(ndt.type(dt), ndt.string)
        # h5py 2.2 style "special dtype"
        dt = np.dtype(('O', [( ({'type': str},'vlen'), 'O' )] ))
        self.assertEqual(ndt.type(dt), ndt.string)
        if sys.version_info < (3, 0):
            dt = np.dtype(('O', [( ({'type': unicode},'vlen'), 'O' )] ))
            self.assertEqual(ndt.type(dt), ndt.string)

    def test_ndt_type_as_numpy(self):
        self.assertEqual(ndt.bool.as_numpy(), np.dtype('bool'))
        self.assertEqual(ndt.int8.as_numpy(), np.dtype('int8'))
        self.assertEqual(ndt.int16.as_numpy(), np.dtype('int16'))
        self.assertEqual(ndt.int32.as_numpy(), np.dtype('int32'))
        self.assertEqual(ndt.int64.as_numpy(), np.dtype('int64'))
        self.assertEqual(ndt.uint8.as_numpy(), np.dtype('uint8'))
        self.assertEqual(ndt.uint16.as_numpy(), np.dtype('uint16'))
        self.assertEqual(ndt.uint32.as_numpy(), np.dtype('uint32'))
        self.assertEqual(ndt.uint64.as_numpy(), np.dtype('uint64'))
        self.assertEqual(ndt.float32.as_numpy(), np.dtype('float32'))
        self.assertEqual(ndt.float64.as_numpy(), np.dtype('float64'))
        self.assertEqual(ndt.complex_float32.as_numpy(), np.dtype('complex64'))
        self.assertEqual(ndt.complex_float64.as_numpy(), np.dtype('complex128'))
        # nonnative byte order
        nonnative = self.nonnative
        self.assertEqual(ndt.make_byteswap(ndt.int16).as_numpy(),
                    np.dtype(nonnative + 'i2'))
        self.assertEqual(ndt.make_byteswap(ndt.float64).as_numpy(),
                    np.dtype(nonnative + 'f8'))
        # aligned struct
        tp0 = ndt.type('c{x : int32, y : int64}').as_numpy()
        tp1 = np.dtype([('x', np.int32), ('y', np.int64)], align=True)
        self.assertEqual(tp0, tp1)
        # unaligned struct
        tp0 = ndt.make_cstruct([ndt.make_unaligned(ndt.int32),
                        ndt.make_unaligned(ndt.int64)],
                        ['x', 'y']).as_numpy()
        tp1 = np.dtype([('x', np.int32), ('y', np.int64)])
        self.assertEqual(tp0, tp1)
        # check some types which can't be converted
        self.assertRaises(TypeError, ndt.date.as_numpy)
        self.assertRaises(TypeError, ndt.datetime.as_numpy)
        self.assertRaises(TypeError, ndt.bytes.as_numpy)
        self.assertRaises(TypeError, ndt.string.as_numpy)

class TestNumpyViewInterop(unittest.TestCase):
    def setUp(self):
        if sys.byteorder == 'little':
            self.nonnative = '>'
        else:
            self.nonnative = '<'

    def test_dynd_scalar_view(self):
        a = np.array(3, dtype='int64')
        n = nd.view(a)
        self.assertEqual(nd.type_of(n), ndt.int64)
        self.assertEqual(nd.as_py(n), 3)
        self.assertEqual(n.access_flags, 'readwrite')
        # Ensure it's a view
        n[...] = 4
        self.assertEqual(a[()], 4)

    def test_dynd_scalar_array(self):
        a = np.array(3, dtype='int64')
        n = nd.array(a)
        self.assertEqual(nd.type_of(n), ndt.int64)
        self.assertEqual(nd.as_py(n), 3)
        self.assertEqual(n.access_flags, 'immutable')
        # Ensure it's not a view
        a[...] = 4
        self.assertEqual(nd.as_py(n), 3)

    def test_dynd_scalar_asarray(self):
        a = np.array(3, dtype='int64')
        n = nd.asarray(a)
        self.assertEqual(nd.type_of(n), ndt.int64)
        self.assertEqual(nd.as_py(n), 3)
        self.assertEqual(n.access_flags, 'readwrite')
        # Ensure it's a view
        n[...] = 4
        self.assertEqual(a[()], 4)

    def test_dynd_view_of_numpy_array(self):
        # Tests viewing a numpy array as a dynd.array
        nonnative = self.nonnative

        a = np.arange(10, dtype=np.int32)
        n = nd.view(a)
        self.assertEqual(nd.dtype_of(n), ndt.int32)
        self.assertEqual(nd.ndim_of(n), a.ndim)
        self.assertEqual(n.shape, a.shape)
        self.assertEqual(n.strides, a.strides)

        a = np.arange(12, dtype=(nonnative + 'i4')).reshape(3,4)
        n = nd.view(a)
        self.assertEqual(nd.dtype_of(n), ndt.make_byteswap(ndt.int32))
        self.assertEqual(nd.ndim_of(n), a.ndim)
        self.assertEqual(n.shape, a.shape)
        self.assertEqual(n.strides, a.strides)

        a = np.arange(49, dtype='i1')
        a = a[1:].view(dtype=np.int32).reshape(4,3)
        n = nd.view(a)
        self.assertEqual(nd.dtype_of(n), ndt.make_unaligned(ndt.int32))
        self.assertEqual(nd.ndim_of(n), a.ndim)
        self.assertEqual(n.shape, a.shape)
        self.assertEqual(n.strides, a.strides)

        a = np.arange(49, dtype='i1')
        a = a[1:].view(dtype=(nonnative + 'i4')).reshape(2,2,3)
        n = nd.view(a)
        self.assertEqual(nd.dtype_of(n),
                ndt.make_unaligned(ndt.make_byteswap(ndt.int32)))
        self.assertEqual(nd.ndim_of(n), a.ndim)
        self.assertEqual(n.shape, a.shape)
        self.assertEqual(n.strides, a.strides)

    def test_numpy_view_of_dynd_array(self):
        # Tests viewing a dynd.array as a numpy array
        nonnative = self.nonnative

        n = nd.range(10, dtype=ndt.int32)
        a = np.asarray(n)
        self.assertEqual(a.dtype, np.dtype(np.int32))
        self.assertTrue(a.flags.aligned)
        self.assertEqual(a.ndim, nd.ndim_of(n))
        self.assertEqual(a.shape, n.shape)
        self.assertEqual(a.strides, n.strides)
        # Make sure it's a view
        a[1] = 100
        self.assertEqual(nd.as_py(n[1]), 100)

        n = nd.view(np.arange(12, dtype=(nonnative + 'i4')).reshape(3,4))
        a = np.asarray(n)
        self.assertEqual(a.dtype, np.dtype(nonnative + 'i4'))
        self.assertTrue(a.flags.aligned)
        self.assertEqual(a.ndim, nd.ndim_of(n))
        self.assertEqual(a.shape, n.shape)
        self.assertEqual(a.strides, n.strides)
        # Make sure it's a view
        a[1,2] = 100
        self.assertEqual(nd.as_py(n[1,2]), 100)

        n = nd.view(np.arange(49, dtype='i1')[1:].view(dtype=np.int32).reshape(4,3))
        a = np.asarray(n)
        self.assertEqual(a.dtype, np.dtype(np.int32))
        self.assertFalse(a.flags.aligned)
        self.assertEqual(a.ndim, nd.ndim_of(n))
        self.assertEqual(a.shape, n.shape)
        self.assertEqual(a.strides, n.strides)
        # Make sure it's a view
        a[1,2] = 100
        self.assertEqual(nd.as_py(n[1,2]), 100)

        n = nd.view(np.arange(49, dtype='i1')[1:].view(
                    dtype=(nonnative + 'i4')).reshape(2,2,3))
        a = np.asarray(n)
        self.assertEqual(a.dtype, np.dtype(nonnative + 'i4'))
        self.assertFalse(a.flags.aligned)
        self.assertEqual(a.ndim, nd.ndim_of(n))
        self.assertEqual(a.shape, n.shape)
        self.assertEqual(a.strides, n.strides)
        # Make sure it's a view
        a[1,1,1] = 100
        self.assertEqual(nd.as_py(n[1,1,1]), 100)

    def test_numpy_view_of_noncontig_dynd_array(self):
        n = nd.range(10)[1::3]
        a = np.asarray(n)
        self.assertEqual(a.dtype, np.dtype('i4'))
        self.assertFalse(a.flags.c_contiguous)
        self.assertEqual(a.ndim, nd.ndim_of(n))
        self.assertEqual(a.shape, n.shape)
        self.assertEqual(a.strides, n.strides)
        # Make sure it's a view as needed
        a[1] = 100
        self.assertEqual(nd.as_py(n[1]), 100)

    def test_numpy_dynd_fixedstring_interop(self):
        # Tests converting fixed-size string arrays to/from numpy
        # ASCII Numpy -> dynd
        a = np.array(['abc', 'testing', 'array'])
        b = nd.view(a)
        if sys.version_info >= (3, 0):
            self.assertEqual(ndt.make_fixedstring(7, 'utf_32'), nd.dtype_of(b))
        else:
            self.assertEqual(ndt.make_fixedstring(7, 'ascii'), nd.dtype_of(b))
        self.assertEqual(nd.dtype_of(b), ndt.type(a.dtype))

        # Make sure it's ascii
        a = a.astype('S7')
        b = nd.view(a)

        # ASCII dynd -> Numpy
        c = np.asarray(b)
        self.assertEqual(a.dtype, c.dtype)
        assert_array_equal(a, c)
        # verify 'a' and 'c' are looking at the same data
        a[1] = 'modify'
        assert_array_equal(a, c)

        # ASCII dynd -> UTF32 dynd
        b_u = b.ucast(ndt.make_fixedstring(7, 'utf_32'))
        self.assertEqual(
                ndt.make_convert(
                    ndt.make_fixedstring(7, 'utf_32'),
                    ndt.make_fixedstring(7, 'ascii')),
                nd.dtype_of(b_u))
        # Evaluate to its value array
        b_u = b_u.eval()
        self.assertEqual(
                ndt.make_fixedstring(7, 'utf_32'),
                nd.dtype_of(b_u))

        # UTF32 dynd -> Numpy
        c_u = np.asarray(b_u)
        self.assertEqual(nd.dtype_of(b_u), ndt.type(c_u.dtype))
        assert_array_equal(a.astype('U'), c_u)
        # 'a' and 'c_u' are not looking at the same data
        a[1] = 'diff'
        self.assertFalse(np.all(a == c_u))

    def test_numpy_blockref_string(self):
        # Blockref strings don't have a corresponding Numpy construct
        # Therefore numpy makes an object array scalar out of them.
        a = nd.array("abcdef")
        self.assertEqual(nd.dtype_of(a), ndt.string)
        # Some versions of NumPy produce an error instead,
        # so this assertion is removed
        #self.assertEqual(np.asarray(a).dtype, np.dtype(object))

        a = nd.array(u"abcdef \uc548\ub155")
        self.assertEqual(nd.dtype_of(a), ndt.string)
        # Some versions of NumPy produce an error instead,
        # so this assertion is removed
        #self.assertEqual(np.asarray(a).dtype, np.dtype(object))

    def test_readwrite_access_flags(self):
        def assign_to(x,y):
            x[0] = y
        # Tests that read/write access control is preserved to/from numpy
        a = np.arange(10.)

        # Writeable
        b = nd.view(a)
        b[0] = 2.0
        self.assertEqual(nd.as_py(b[0]), 2.0)
        self.assertEqual(a[0], 2.0)

        # Readonly view of writeable
        b = nd.view(a, access='r')
        self.assertRaises(RuntimeError, assign_to, b, 3.0)
        # should still be 2.0
        self.assertEqual(nd.as_py(b[0]), 2.0)
        self.assertEqual(a[0], 2.0)

        # Not writeable
        a.flags.writeable = False
        b = nd.view(a)
        self.assertRaises(RuntimeError, assign_to, b, 3.0)
        # should still be 2.0
        self.assertEqual(nd.as_py(b[0]), 2.0)
        self.assertEqual(a[0], 2.0)
        # Trying to get a readwrite view raises an error
        self.assertRaises(RuntimeError, nd.view, a, access='rw')

class TestAsNumpy(unittest.TestCase):
    def test_struct_as_numpy(self):
        # Aligned cstruct
        a = nd.array([[1, 2], [3, 4]], dtype='{x : int32, y: int64}')
        b = nd.as_numpy(a)
        self.assertEqual(b.dtype,
                    np.dtype([('x', np.int32), ('y', np.int64)], align=True))
        self.assertEqual(nd.as_py(a.x), b['x'].tolist())
        self.assertEqual(nd.as_py(a.y), b['y'].tolist())
        # Unaligned cstruct
        a = nd.array([[1, 2], [3, 4]],
                    dtype='{x : unaligned[int32], y: unaligned[int64]}')
        b = nd.as_numpy(a)
        self.assertEqual(b.dtype, np.dtype([('x', np.int32), ('y', np.int64)]))
        self.assertEqual(nd.as_py(a.x), b['x'].tolist())
        self.assertEqual(nd.as_py(a.y), b['y'].tolist())

    def test_cstruct_as_numpy(self):
        # Aligned cstruct
        a = nd.array([[1, 2], [3, 4]], dtype='c{x : int32, y: int64}')
        b = nd.as_numpy(a)
        self.assertEqual(b.dtype,
                    np.dtype([('x', np.int32), ('y', np.int64)], align=True))
        self.assertEqual(nd.as_py(a.x), b['x'].tolist())
        self.assertEqual(nd.as_py(a.y), b['y'].tolist())
        # Unaligned cstruct
        a = nd.array([[1, 2], [3, 4]],
                    dtype='c{x : unaligned[int32], y: unaligned[int64]}')
        b = nd.as_numpy(a)
        self.assertEqual(b.dtype, np.dtype([('x', np.int32), ('y', np.int64)]))
        self.assertEqual(nd.as_py(a.x), b['x'].tolist())
        self.assertEqual(nd.as_py(a.y), b['y'].tolist())

    def test_cstruct_via_pep3118(self):
        # Aligned cstruct
        a = nd.array([[1, 2], [3, 4]], dtype='c{x : int32, y: int64}')
        b = np.asarray(a)
        self.assertEqual(b.dtype,
                    np.dtype([('x', np.int32), ('y', np.int64)], align=True))
        self.assertEqual(nd.as_py(a.x), b['x'].tolist())
        self.assertEqual(nd.as_py(a.y), b['y'].tolist())
        # Unaligned cstruct
        a = nd.array([[1, 2], [3, 4]],
                    dtype='c{x : unaligned[int32], y: unaligned[int64]}')
        b = np.asarray(a)
        self.assertEqual(b.dtype, np.dtype([('x', np.int32), ('y', np.int64)]))
        self.assertEqual(nd.as_py(a.x), b['x'].tolist())
        self.assertEqual(nd.as_py(a.y), b['y'].tolist())

    def test_fixed_dim(self):
        a = nd.array([1, 3, 5], type='3 * int32')
        b = nd.as_numpy(a)
        self.assertEqual(b.dtype, np.dtype('int32'))
        self.assertEqual(b.tolist(), [1, 3, 5])

    def test_fixed_dim_via_pep3118(self):
        a = nd.array([1, 3, 5], type='3 * int32')
        b = np.asarray(a)
        self.assertEqual(b.dtype, np.dtype('int32'))
        self.assertEqual(b.tolist(), [1, 3, 5])

class TestNumpyScalarInterop(unittest.TestCase):
    def test_numpy_scalar_conversion_dtypes(self):
        self.assertEqual(nd.dtype_of(nd.array(np.bool_(True))), ndt.bool)
        self.assertEqual(nd.dtype_of(nd.array(np.bool(True))), ndt.bool)
        self.assertEqual(nd.dtype_of(nd.array(np.int8(100))), ndt.int8)
        self.assertEqual(nd.dtype_of(nd.array(np.int16(100))), ndt.int16)
        self.assertEqual(nd.dtype_of(nd.array(np.int32(100))), ndt.int32)
        self.assertEqual(nd.dtype_of(nd.array(np.int64(100))), ndt.int64)
        self.assertEqual(nd.dtype_of(nd.array(np.uint8(100))), ndt.uint8)
        self.assertEqual(nd.dtype_of(nd.array(np.uint16(100))), ndt.uint16)
        self.assertEqual(nd.dtype_of(nd.array(np.uint32(100))), ndt.uint32)
        self.assertEqual(nd.dtype_of(nd.array(np.uint64(100))), ndt.uint64)
        self.assertEqual(nd.dtype_of(nd.array(np.float32(100.))), ndt.float32)
        self.assertEqual(nd.dtype_of(nd.array(np.float64(100.))), ndt.float64)
        self.assertEqual(nd.dtype_of(nd.array(np.complex64(100j))),
                         ndt.complex_float32)
        self.assertEqual(nd.dtype_of(nd.array(np.complex128(100j))),
                         ndt.complex_float64)
        if np.__version__ >= '1.7':
            self.assertEqual(nd.dtype_of(nd.array(np.datetime64('2000-12-13'))),
                             ndt.date)
            self.assertEqual(nd.dtype_of(nd.array(np.datetime64('2000-12-13T12:30'))),
                             ndt.type('datetime[tz="UTC"]'))

    def test_numpy_scalar_conversion_values(self):
        self.assertEqual(nd.as_py(nd.array(np.bool_(True))), True)
        self.assertEqual(nd.as_py(nd.array(np.bool_(False))), False)
        self.assertEqual(nd.as_py(nd.array(np.int8(100))), 100)
        self.assertEqual(nd.as_py(nd.array(np.int8(-100))), -100)
        self.assertEqual(nd.as_py(nd.array(np.int16(20000))), 20000)
        self.assertEqual(nd.as_py(nd.array(np.int16(-20000))), -20000)
        self.assertEqual(nd.as_py(nd.array(np.int32(1000000000))), 1000000000)
        self.assertEqual(nd.as_py(nd.array(np.int64(-1000000000000))),
                         -1000000000000)
        self.assertEqual(nd.as_py(nd.array(np.int64(1000000000000))),
                         1000000000000)
        self.assertEqual(nd.as_py(nd.array(np.int32(-1000000000))),
                         -1000000000)
        self.assertEqual(nd.as_py(nd.array(np.uint8(200))), 200)
        self.assertEqual(nd.as_py(nd.array(np.uint16(50000))), 50000)
        self.assertEqual(nd.as_py(nd.array(np.uint32(3000000000))), 3000000000)
        self.assertEqual(nd.as_py(nd.array(np.uint64(10000000000000000000))),
                         10000000000000000000)
        self.assertEqual(nd.as_py(nd.array(np.float32(2.5))), 2.5)
        self.assertEqual(nd.as_py(nd.array(np.float64(2.5))), 2.5)
        self.assertEqual(nd.as_py(nd.array(np.complex64(2.5-1j))), 2.5-1j)
        self.assertEqual(nd.as_py(nd.array(np.complex128(2.5-1j))), 2.5-1j)
        if np.__version__ >= '1.7':
            # Various date units
            self.assertEqual(nd.as_py(nd.array(np.datetime64('2000'))),
                             date(2000, 1, 1))
            self.assertEqual(nd.as_py(nd.array(np.datetime64('2000-12'))),
                             date(2000, 12, 1))
            self.assertEqual(nd.as_py(nd.array(np.datetime64('2000-12-13'))),
                             date(2000, 12, 13))
            # Various datetime units
            self.assertEqual(nd.as_py(nd.array(np.datetime64('2000-12-13T12Z'))),
                             datetime(2000, 12, 13, 12, 0))
            self.assertEqual(nd.as_py(nd.array(np.datetime64('2000-12-13T12:30Z'))),
                             datetime(2000, 12, 13, 12, 30))
            self.assertEqual(nd.as_py(nd.array(np.datetime64('1823-12-13T12:30Z'))),
                             datetime(1823, 12, 13, 12, 30))
            self.assertEqual(nd.as_py(nd.array(np.datetime64('2000-12-13T12:30:24Z'))),
                             datetime(2000, 12, 13, 12, 30, 24))
            self.assertEqual(nd.as_py(nd.array(np.datetime64('2000-12-13T12:30:24.123Z'))),
                             datetime(2000, 12, 13, 12, 30, 24, 123000))
            self.assertEqual(nd.as_py(nd.array(np.datetime64('2000-12-13T12:30:24.123456Z'))),
                             datetime(2000, 12, 13, 12, 30, 24, 123456))
            self.assertEqual(nd.as_py(nd.array(np.datetime64('2000-12-13T12:30:24.123456124Z'))),
                             datetime(2000, 12, 13, 12, 30, 24, 123456))
            self.assertEqual(str(nd.array(np.datetime64('2000-12-13T12:30:24.123456124Z'))),
                             '2000-12-13T12:30:24.1234561Z')
            self.assertEqual(str(nd.array(np.datetime64('1842-12-13T12:30:24.123456124Z'))),
                             '1842-12-13T12:30:24.1234561Z')

    def test_numpy_struct_scalar(self):
        # Create a NumPy struct scalar object, by indexing into
        # a structured array
        a = np.array([(10, 11, 12)], dtype='i4,i8,f8')[0]
        aligned_tp = ndt.type('c{f0: int32, f1: int64, f2: float64}')
        val = {'f0': 10, 'f1': 11, 'f2': 12}

        # Construct using nd.array
        b = nd.array(a)
        self.assertEqual(nd.type_of(b), aligned_tp)
        self.assertEqual(nd.as_py(b), val)
        self.assertEqual(b.access_flags, 'immutable')
        b = nd.array(a, access='rw')
        self.assertEqual(nd.type_of(b), aligned_tp)
        self.assertEqual(nd.as_py(b), val)
        self.assertEqual(b.access_flags, 'readwrite')

        # Construct using nd.asarray
        b = nd.asarray(a)
        self.assertEqual(nd.type_of(b), aligned_tp)
        self.assertEqual(nd.as_py(b), val)
        self.assertEqual(b.access_flags, 'immutable')
        b = nd.asarray(a, access='rw')
        self.assertEqual(nd.type_of(b), aligned_tp)
        self.assertEqual(nd.as_py(b), val)
        self.assertEqual(b.access_flags, 'readwrite')

        # nd.view should fail
        self.assertRaises(RuntimeError, nd.view, a)

    def test_expr_struct_conversion(self):
        a = nd.array([date(2000, 12, 13), date(1995, 5, 2)]).to_struct()
        b = nd.as_numpy(a, allow_copy=True)
        self.assertTrue(isinstance(b, np.ndarray))
        # Use the NumPy assertions which support arrays
        assert_equal(b['year'], [2000, 1995])
        assert_equal(b['month'], [12, 5])
        assert_equal(b['day'], [13, 2])

    def test_var_dim_conversion(self):
        # A simple instantiated var_dim array should be
        # viewable with numpy without changes
        a = nd.array([1, 2, 3, 4, 5], type='var * int32')
        b = nd.as_numpy(a)
        self.assertTrue(isinstance(b, np.ndarray))
        self.assertEqual(b.dtype, np.dtype('int32'))
        # Use the NumPy assertions which support arrays
        assert_equal(b, [1, 2, 3, 4, 5])

    def test_date_from_numpy(self):
        a = np.array(['2000-12-13', '1995-05-02'], dtype='M8[D]')
        b = nd.array(a)
        self.assertEqual(nd.type_of(b), ndt.type('strided * date'))
        self.assertEqual(nd.as_py(b), [date(2000, 12, 13), date(1995, 5, 2)])

    def test_date_as_numpy(self):
        a = nd.array([date(2000, 12, 13), date(1995, 5, 2)])
        b = nd.as_numpy(a, allow_copy=True)
        assert_equal(b, np.array(['2000-12-13', '1995-05-02'], dtype='M8[D]'))

    def test_datetime_from_numpy(self):
        # NumPy hours unit
        a = np.array(['2000-12-13T12Z', '1955-05-02T02Z'],
                     dtype='M8[h]')
        b = nd.array(a)
        self.assertEqual(nd.type_of(b), ndt.type('strided * datetime[tz="UTC"]'))
        self.assertEqual(nd.as_py(b), [datetime(2000, 12, 13, 12),
                                   datetime(1955, 5, 2, 2)])
        # NumPy minutes unit
        a = np.array(['2000-12-13T12:30Z', '1955-05-02T02:23Z'],
                     dtype='M8[m]')
        b = nd.array(a)
        self.assertEqual(nd.type_of(b), ndt.type('strided * datetime[tz="UTC"]'))
        self.assertEqual(nd.as_py(b), [datetime(2000, 12, 13, 12, 30),
                                   datetime(1955, 5, 2, 2, 23)])
        # NumPy seconds unit
        a = np.array(['2000-12-13T12:30:51Z', '1955-05-02T02:23:29Z'],
                     dtype='M8[s]')
        b = nd.array(a)
        self.assertEqual(nd.type_of(b), ndt.type('strided * datetime[tz="UTC"]'))
        self.assertEqual(nd.as_py(b), [datetime(2000, 12, 13, 12, 30, 51),
                                   datetime(1955, 5, 2, 2, 23, 29)])
        # NumPy milliseconds unit
        a = np.array(['2000-12-13T12:30:51.123Z', '1955-05-02T02:23:29.456Z'],
                     dtype='M8[ms]')
        b = nd.array(a)
        self.assertEqual(nd.type_of(b), ndt.type('strided * datetime[tz="UTC"]'))
        self.assertEqual(nd.as_py(b), [datetime(2000, 12, 13, 12, 30, 51, 123000),
                                   datetime(1955, 5, 2, 2, 23, 29, 456000)])
        # NumPy microseconds unit
        a = np.array(['2000-12-13T12:30:51.123456Z', '1955-05-02T02:23:29.456123Z'],
                     dtype='M8[us]')
        b = nd.array(a)
        self.assertEqual(nd.type_of(b), ndt.type('strided * datetime[tz="UTC"]'))
        self.assertEqual(nd.as_py(b), [datetime(2000, 12, 13, 12, 30, 51, 123456),
                                   datetime(1955, 5, 2, 2, 23, 29, 456123)])
        # NumPy nanoseconds unit (truncated to 100 nanosecond ticks)
        a = np.array(['2000-12-13T12:30:51.123456987Z',
                      '1955-05-02T02:23:29.456123798Z'],
                     dtype='M8[ns]')
        b = nd.array(a)
        self.assertEqual(nd.type_of(b),
                         ndt.type('strided * datetime[tz="UTC"]'))
        self.assertEqual([str(x) for x in b], ["2000-12-13T12:30:51.1234569Z",
                                               "1955-05-02T02:23:29.4561237Z"])

    def test_misaligned_datetime_from_numpy(self):
        a = np.array([(1, "2000-12-25T00:00:01Z"),
                      (2, "2001-12-25T00:00:01Z"),
                      (3, "2002-12-25T00:00:01Z")],
                     dtype=[('id', 'int8'), ('ts', 'M8[us]')])
        b = nd.view(a)
        self.assertEqual(nd.type_of(b),
            ndt.type("strided * c{id : int8, ts: adapt[(unaligned[int64]) -> datetime[tz='UTC'], 'microseconds since 1970']}"))
        self.assertEqual(nd.as_py(b, tuple=True),
                         [(1, datetime(2000, 12, 25, 0, 0, 1)),
                          (2, datetime(2001, 12, 25, 0, 0, 1)),
                          (3, datetime(2002, 12, 25, 0, 0, 1))])

    def test_datetime_as_numpy(self):
        a = nd.array(['2000-12-13T12:30',
                      '1995-05-02T2:15:33'],
                     dtype='datetime[tz="UTC"]')
        b = nd.as_numpy(a, allow_copy=True)
        assert_equal(b, np.array(['2000-12-13T12:30Z', '1995-05-02T02:15:33Z'],
                                 dtype='M8[us]'))
    def test_string_as_numpy(self):
        a = nd.array(["this", "is", "a", "test of varlen strings"])
        b = nd.as_numpy(a, allow_copy=True)
        self.assertEqual(b.dtype, np.dtype('O'))
        assert_equal(b, np.array(["this", "is", "a", "test of varlen strings"],
                                 dtype='O'))
        # Also in a struct
        a = nd.array([(1, "testing", 1.5), (10, "abc", 2)],
                     type="strided * {x: int, y: string, z: real}")
        b = nd.as_numpy(a, allow_copy=True)
        self.assertEqual(b.dtype, np.dtype([('x', 'int32'),
                                            ('y', 'O'),
                                            ('z', 'float64')], align=True))
        self.assertEqual(b.tolist(), [(1, "testing", 1.5), (10, "abc", 2)])


if __name__ == '__main__':
    unittest.main()

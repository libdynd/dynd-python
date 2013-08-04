import sys
import unittest
from dynd import nd, ndt
import numpy as np
from datetime import date
from numpy.testing import *

class TestNumpyDTypeInterop(unittest.TestCase):
    def setUp(self):
        if sys.byteorder == 'little':
            self.nonnative = '>'
        else:
            self.nonnative = '<'

    def test_dynd_type_from_numpy_scalar_types(self):
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
        self.assertEqual(ndt.cfloat32, ndt.type(np.complex64))
        self.assertEqual(ndt.cfloat64, ndt.type(np.complex128))

    def test_dynd_type_from_numpy_dtype(self):
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
        self.assertEqual(ndt.cfloat32, ndt.type(np.dtype(np.complex64)))
        self.assertEqual(ndt.cfloat64, ndt.type(np.dtype(np.complex128)))
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
        self.assertEqual(ndt.make_byteswap(ndt.cfloat32),
                ndt.type(np.dtype(nonnative + 'c8')))
        self.assertEqual(ndt.make_byteswap(ndt.cfloat64),
                ndt.type(np.dtype(nonnative + 'c16')))

class TestNumpyViewInterop(unittest.TestCase):
    def setUp(self):
        if sys.byteorder == 'little':
            self.nonnative = '>'
        else:
            self.nonnative = '<'

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

        n = nd.view(np.arange(10, dtype=np.int32))
        a = np.asarray(n)
        self.assertEqual(a.dtype, np.dtype(np.int32))
        self.assertTrue(a.flags.aligned)
        self.assertEqual(a.ndim, nd.ndim_of(n))
        self.assertEqual(a.shape, n.shape)
        self.assertEqual(a.strides, n.strides)

        n = nd.view(np.arange(12, dtype=(nonnative + 'i4')).reshape(3,4))
        a = np.asarray(n)
        self.assertEqual(a.dtype, np.dtype(nonnative + 'i4'))
        self.assertTrue(a.flags.aligned)
        self.assertEqual(a.ndim, nd.ndim_of(n))
        self.assertEqual(a.shape, n.shape)
        self.assertEqual(a.strides, n.strides)

        n = nd.view(np.arange(49, dtype='i1')[1:].view(dtype=np.int32).reshape(4,3))
        a = np.asarray(n)
        self.assertEqual(a.dtype, np.dtype(np.int32))
        self.assertFalse(a.flags.aligned)
        self.assertEqual(a.ndim, nd.ndim_of(n))
        self.assertEqual(a.shape, n.shape)
        self.assertEqual(a.strides, n.strides)

        n = nd.view(np.arange(49, dtype='i1')[1:].view(
                    dtype=(nonnative + 'i4')).reshape(2,2,3))
        a = np.asarray(n)
        self.assertEqual(a.dtype, np.dtype(nonnative + 'i4'))
        self.assertFalse(a.flags.aligned)
        self.assertEqual(a.ndim, nd.ndim_of(n))
        self.assertEqual(a.shape, n.shape)
        self.assertEqual(a.strides, n.strides)

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
        self.assertEqual(nd.dtype_of(nd.array(np.complex64(100j))), ndt.cfloat32)
        self.assertEqual(nd.dtype_of(nd.array(np.complex128(100j))), ndt.cfloat64)
        if np.__version__ >= '1.7':
            self.assertEqual(nd.dtype_of(nd.array(np.datetime64('2000-12-13'))), ndt.date)

    def test_numpy_scalar_conversion_values(self):
        self.assertEqual(nd.as_py(nd.array(np.bool_(True))), True)
        self.assertEqual(nd.as_py(nd.array(np.bool_(False))), False)
        self.assertEqual(nd.as_py(nd.array(np.int8(100))), 100)
        self.assertEqual(nd.as_py(nd.array(np.int8(-100))), -100)
        self.assertEqual(nd.as_py(nd.array(np.int16(20000))), 20000)
        self.assertEqual(nd.as_py(nd.array(np.int16(-20000))), -20000)
        self.assertEqual(nd.as_py(nd.array(np.int32(1000000000))), 1000000000)
        self.assertEqual(nd.as_py(nd.array(np.int64(-1000000000000))), -1000000000000)
        self.assertEqual(nd.as_py(nd.array(np.int64(1000000000000))), 1000000000000)
        self.assertEqual(nd.as_py(nd.array(np.int32(-1000000000))), -1000000000)
        self.assertEqual(nd.as_py(nd.array(np.uint8(200))), 200)
        self.assertEqual(nd.as_py(nd.array(np.uint16(50000))), 50000)
        self.assertEqual(nd.as_py(nd.array(np.uint32(3000000000))), 3000000000)
        self.assertEqual(nd.as_py(nd.array(np.uint64(10000000000000000000))), 10000000000000000000)
        self.assertEqual(nd.as_py(nd.array(np.float32(2.5))), 2.5)
        self.assertEqual(nd.as_py(nd.array(np.float64(2.5))), 2.5)
        self.assertEqual(nd.as_py(nd.array(np.complex64(2.5-1j))), 2.5-1j)
        self.assertEqual(nd.as_py(nd.array(np.complex128(2.5-1j))), 2.5-1j)
        if np.__version__ >= '1.7':
            self.assertEqual(nd.as_py(nd.array(np.datetime64('2000-12-13'))), date(2000, 12, 13))

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
        # vieable with numpy without changes
        a = nd.array([1, 2, 3, 4, 5], type='var, int32')
        b = nd.as_numpy(a)
        self.assertTrue(isinstance(b, np.ndarray))
        self.assertEqual(b.dtype, np.dtype('int32'))
        # Use the NumPy assertions which support arrays
        assert_equal(b, [1, 2, 3, 4, 5])

if __name__ == '__main__':
    unittest.main()

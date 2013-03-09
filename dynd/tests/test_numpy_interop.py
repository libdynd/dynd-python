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

    def test_dtype_from_numpy_scalar_types(self):
        # Tests converting numpy scalar types to pydynd dtypes
        self.assertEqual(ndt.bool, nd.dtype(np.bool))
        self.assertEqual(ndt.bool, nd.dtype(np.bool_))
        self.assertEqual(ndt.int8, nd.dtype(np.int8))
        self.assertEqual(ndt.int16, nd.dtype(np.int16))
        self.assertEqual(ndt.int32, nd.dtype(np.int32))
        self.assertEqual(ndt.int64, nd.dtype(np.int64))
        self.assertEqual(ndt.uint8, nd.dtype(np.uint8))
        self.assertEqual(ndt.uint16, nd.dtype(np.uint16))
        self.assertEqual(ndt.uint32, nd.dtype(np.uint32))
        self.assertEqual(ndt.uint64, nd.dtype(np.uint64))
        self.assertEqual(ndt.float32, nd.dtype(np.float32))
        self.assertEqual(ndt.float64, nd.dtype(np.float64))
        self.assertEqual(ndt.cfloat32, nd.dtype(np.complex64))
        self.assertEqual(ndt.cfloat64, nd.dtype(np.complex128))

    def test_dtype_from_numpy_dtype(self):
        # Tests converting numpy dtypes to pydynd dtypes
        # native byte order
        self.assertEqual(ndt.bool, nd.dtype(np.dtype(np.bool)))
        self.assertEqual(ndt.int8, nd.dtype(np.dtype(np.int8)))
        self.assertEqual(ndt.int16, nd.dtype(np.dtype(np.int16)))
        self.assertEqual(ndt.int32, nd.dtype(np.dtype(np.int32)))
        self.assertEqual(ndt.int64, nd.dtype(np.dtype(np.int64)))
        self.assertEqual(ndt.uint8, nd.dtype(np.dtype(np.uint8)))
        self.assertEqual(ndt.uint16, nd.dtype(np.dtype(np.uint16)))
        self.assertEqual(ndt.uint32, nd.dtype(np.dtype(np.uint32)))
        self.assertEqual(ndt.uint64, nd.dtype(np.dtype(np.uint64)))
        self.assertEqual(ndt.float32, nd.dtype(np.dtype(np.float32)))
        self.assertEqual(ndt.float64, nd.dtype(np.dtype(np.float64)))
        self.assertEqual(ndt.cfloat32, nd.dtype(np.dtype(np.complex64)))
        self.assertEqual(ndt.cfloat64, nd.dtype(np.dtype(np.complex128)))
        self.assertEqual(ndt.make_fixedstring_dtype(10, 'ascii'),
                    nd.dtype(np.dtype('S10')))
        self.assertEqual(ndt.make_fixedstring_dtype(10, 'utf_32'),
                    nd.dtype(np.dtype('U10')))

        # non-native byte order
        nonnative = self.nonnative

        self.assertEqual(ndt.make_byteswap_dtype(ndt.int16),
                nd.dtype(np.dtype(nonnative + 'i2')))
        self.assertEqual(ndt.make_byteswap_dtype(ndt.int32),
                nd.dtype(np.dtype(nonnative + 'i4')))
        self.assertEqual(ndt.make_byteswap_dtype(ndt.int64),
                nd.dtype(np.dtype(nonnative + 'i8')))
        self.assertEqual(ndt.make_byteswap_dtype(ndt.uint16),
                nd.dtype(np.dtype(nonnative + 'u2')))
        self.assertEqual(ndt.make_byteswap_dtype(ndt.uint32),
                nd.dtype(np.dtype(nonnative + 'u4')))
        self.assertEqual(ndt.make_byteswap_dtype(ndt.uint64),
                nd.dtype(np.dtype(nonnative + 'u8')))
        self.assertEqual(ndt.make_byteswap_dtype(ndt.float32),
                nd.dtype(np.dtype(nonnative + 'f4')))
        self.assertEqual(ndt.make_byteswap_dtype(ndt.float64),
                nd.dtype(np.dtype(nonnative + 'f8')))
        self.assertEqual(ndt.make_byteswap_dtype(ndt.cfloat32),
                nd.dtype(np.dtype(nonnative + 'c8')))
        self.assertEqual(ndt.make_byteswap_dtype(ndt.cfloat64),
                nd.dtype(np.dtype(nonnative + 'c16')))

class TestNumpyViewInterop(unittest.TestCase):
    def setUp(self):
        if sys.byteorder == 'little':
            self.nonnative = '>'
        else:
            self.nonnative = '<'

    def test_dynd_view_of_numpy_array(self):
        # Tests viewing a numpy array as a dynd.ndobject
        nonnative = self.nonnative

        a = np.arange(10, dtype=np.int32)
        n = nd.ndobject(a)
        self.assertEqual(n.udtype, ndt.int32)
        self.assertEqual(n.undim, a.ndim)
        self.assertEqual(n.shape, a.shape)
        self.assertEqual(n.strides, a.strides)

        a = np.arange(12, dtype=(nonnative + 'i4')).reshape(3,4)
        n = nd.ndobject(a)
        self.assertEqual(n.udtype, ndt.make_byteswap_dtype(ndt.int32))
        self.assertEqual(n.undim, a.ndim)
        self.assertEqual(n.shape, a.shape)
        self.assertEqual(n.strides, a.strides)

        a = np.arange(49, dtype='i1')
        a = a[1:].view(dtype=np.int32).reshape(4,3)
        n = nd.ndobject(a)
        self.assertEqual(n.udtype, ndt.make_unaligned_dtype(ndt.int32))
        self.assertEqual(n.undim, a.ndim)
        self.assertEqual(n.shape, a.shape)
        self.assertEqual(n.strides, a.strides)

        a = np.arange(49, dtype='i1')
        a = a[1:].view(dtype=(nonnative + 'i4')).reshape(2,2,3)
        n = nd.ndobject(a)
        self.assertEqual(n.udtype,
                ndt.make_unaligned_dtype(ndt.make_byteswap_dtype(ndt.int32)))
        self.assertEqual(n.undim, a.ndim)
        self.assertEqual(n.shape, a.shape)
        self.assertEqual(n.strides, a.strides)

    def test_numpy_view_of_dynd_array(self):
        # Tests viewing a dynd.ndobject as a numpy array
        nonnative = self.nonnative

        n = nd.ndobject(np.arange(10, dtype=np.int32))
        a = np.asarray(n)
        self.assertEqual(a.dtype, np.dtype(np.int32))
        self.assertTrue(a.flags.aligned)
        self.assertEqual(a.ndim, n.undim)
        self.assertEqual(a.shape, n.shape)
        self.assertEqual(a.strides, n.strides)

        n = nd.ndobject(np.arange(12, dtype=(nonnative + 'i4')).reshape(3,4))
        a = np.asarray(n)
        self.assertEqual(a.dtype, np.dtype(nonnative + 'i4'))
        self.assertTrue(a.flags.aligned)
        self.assertEqual(a.ndim, n.undim)
        self.assertEqual(a.shape, n.shape)
        self.assertEqual(a.strides, n.strides)

        n = nd.ndobject(np.arange(49, dtype='i1')[1:].view(dtype=np.int32).reshape(4,3))
        a = np.asarray(n)
        self.assertEqual(a.dtype, np.dtype(np.int32))
        self.assertFalse(a.flags.aligned)
        self.assertEqual(a.ndim, n.undim)
        self.assertEqual(a.shape, n.shape)
        self.assertEqual(a.strides, n.strides)

        n = nd.ndobject(np.arange(49, dtype='i1')[1:].view(
                    dtype=(nonnative + 'i4')).reshape(2,2,3))
        a = np.asarray(n)
        self.assertEqual(a.dtype, np.dtype(nonnative + 'i4'))
        self.assertFalse(a.flags.aligned)
        self.assertEqual(a.ndim, n.undim)
        self.assertEqual(a.shape, n.shape)
        self.assertEqual(a.strides, n.strides)

    def test_numpy_dynd_fixedstring_interop(self):
        # Tests converting fixed-size string arrays to/from numpy
        # ASCII Numpy -> dynd
        a = np.array(['abc', 'testing', 'array'])
        b = nd.ndobject(a)
        self.assertEqual(ndt.make_fixedstring_dtype(7, 'ascii'), b.udtype)
        self.assertEqual(b.udtype, nd.dtype(a.dtype))

        # ASCII dynd -> Numpy
        c = np.asarray(b)
        self.assertEqual(a.dtype, c.dtype)
        assert_array_equal(a, c)
        # verify 'a' and 'c' are looking at the same data
        a[1] = 'modify'
        assert_array_equal(a, c)

        # ASCII dynd -> UTF32 dynd
        b_u = b.ucast(ndt.make_fixedstring_dtype(7, 'utf_32'))
        self.assertEqual(
                ndt.make_convert_dtype(
                    ndt.make_fixedstring_dtype(7, 'utf_32'),
                    ndt.make_fixedstring_dtype(7, 'ascii')),
                b_u.udtype)
        # Evaluate to its value array
        b_u = b_u.eval()
        self.assertEqual(
                ndt.make_fixedstring_dtype(7, 'utf_32'),
                b_u.udtype)

        # UTF32 dynd -> Numpy
        c_u = np.asarray(b_u)
        self.assertEqual(b_u.udtype, nd.dtype(c_u.dtype))
        assert_array_equal(a, c_u)
        # 'a' and 'c_u' are not looking at the same data
        a[1] = 'diff'
        self.assertFalse(np.all(a == c_u))

    def test_numpy_blockref_string(self):
        # Blockref strings don't have a corresponding Numpy construct
        # Therefore numpy makes an object array scalar out of them.
        a = nd.ndobject("abcdef")
        self.assertEqual(
                ndt.make_string_dtype('ascii'),
                a.dtype)
        # Some versions of NumPy produce an error instead,
        # so this assertion is removed
        #self.assertEqual(np.asarray(a).dtype, np.dtype(object))

        a = nd.ndobject(u"abcdef \uc548\ub155")
        self.assert_(a.dtype in [
                ndt.make_string_dtype('ucs_2'),
                ndt.make_string_dtype('utf_32')])
        # Some versions of NumPy produce an error instead,
        # so this assertion is removed
        #self.assertEqual(np.asarray(a).dtype, np.dtype(object))

    def test_readwrite_access_flags(self):
        # Tests that read/write access control is preserved to/from numpy
        a = np.arange(10.)

        # Writeable
        b = nd.ndobject(a)
        b[0] = 2.0
        self.assertEqual(nd.as_py(b[0]), 2.0)
        self.assertEqual(a[0], 2.0)

        # Not writeable
        a.flags.writeable = False
        b = nd.ndobject(a)
        def assign_to(x,y):
            x[0] = y
        self.assertRaises(RuntimeError, assign_to, b, 3.0)
        # should still be 2.0
        self.assertEqual(nd.as_py(b[0]), 2.0)
        self.assertEqual(a[0], 2.0)

class TestNumpyScalarInterop(unittest.TestCase):
    def test_numpy_scalar_conversion_dtypes(self):
        self.assertEqual(nd.ndobject(np.bool_(True)).dtype, ndt.bool)
        self.assertEqual(nd.ndobject(np.bool(True)).dtype, ndt.bool)
        self.assertEqual(nd.ndobject(np.int8(100)).dtype, ndt.int8)
        self.assertEqual(nd.ndobject(np.int16(100)).dtype, ndt.int16)
        self.assertEqual(nd.ndobject(np.int32(100)).dtype, ndt.int32)
        self.assertEqual(nd.ndobject(np.int64(100)).dtype, ndt.int64)
        self.assertEqual(nd.ndobject(np.uint8(100)).dtype, ndt.uint8)
        self.assertEqual(nd.ndobject(np.uint16(100)).dtype, ndt.uint16)
        self.assertEqual(nd.ndobject(np.uint32(100)).dtype, ndt.uint32)
        self.assertEqual(nd.ndobject(np.uint64(100)).dtype, ndt.uint64)
        self.assertEqual(nd.ndobject(np.float32(100.)).dtype, ndt.float32)
        self.assertEqual(nd.ndobject(np.float64(100.)).dtype, ndt.float64)
        self.assertEqual(nd.ndobject(np.complex64(100j)).dtype, ndt.cfloat32)
        self.assertEqual(nd.ndobject(np.complex128(100j)).dtype, ndt.cfloat64)
        if np.__version__ >= '1.7':
            self.assertEqual(nd.ndobject(np.datetime64('2000-12-13')).dtype, ndt.date)

    def test_numpy_scalar_conversion_values(self):
        self.assertEqual(nd.as_py(nd.ndobject(np.bool_(True))), True)
        self.assertEqual(nd.as_py(nd.ndobject(np.bool_(False))), False)
        self.assertEqual(nd.as_py(nd.ndobject(np.int8(100))), 100)
        self.assertEqual(nd.as_py(nd.ndobject(np.int8(-100))), -100)
        self.assertEqual(nd.as_py(nd.ndobject(np.int16(20000))), 20000)
        self.assertEqual(nd.as_py(nd.ndobject(np.int16(-20000))), -20000)
        self.assertEqual(nd.as_py(nd.ndobject(np.int32(1000000000))), 1000000000)
        self.assertEqual(nd.as_py(nd.ndobject(np.int64(-1000000000000))), -1000000000000)
        self.assertEqual(nd.as_py(nd.ndobject(np.int64(1000000000000))), 1000000000000)
        self.assertEqual(nd.as_py(nd.ndobject(np.int32(-1000000000))), -1000000000)
        self.assertEqual(nd.as_py(nd.ndobject(np.uint8(200))), 200)
        self.assertEqual(nd.as_py(nd.ndobject(np.uint16(50000))), 50000)
        self.assertEqual(nd.as_py(nd.ndobject(np.uint32(3000000000))), 3000000000)
        self.assertEqual(nd.as_py(nd.ndobject(np.uint64(10000000000000000000))), 10000000000000000000)
        self.assertEqual(nd.as_py(nd.ndobject(np.float32(2.5))), 2.5)
        self.assertEqual(nd.as_py(nd.ndobject(np.float64(2.5))), 2.5)
        self.assertEqual(nd.as_py(nd.ndobject(np.complex64(2.5-1j))), 2.5-1j)
        self.assertEqual(nd.as_py(nd.ndobject(np.complex128(2.5-1j))), 2.5-1j)
        if np.__version__ >= '1.7':
            self.assertEqual(nd.as_py(nd.ndobject(np.datetime64('2000-12-13'))), date(2000, 12, 13))

    def test_expr_struct_conversion(self):
        a = nd.ndobject([date(2000, 12, 13), date(1995, 5, 2)]).to_struct()
        b = nd.as_numpy(a, allow_copy=True)
        # Use the NumPy assertions which support arrays
        assert_equal(b['year'], [2000, 1995])
        assert_equal(b['month'], [12, 5])
        assert_equal(b['day'], [13, 2])

if __name__ == '__main__':
    unittest.main()

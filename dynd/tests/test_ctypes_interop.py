import sys
import unittest
from dynd import nd, ndt
import ctypes

class TestCTypesDTypeInterop(unittest.TestCase):
    def test_dtype_from_ctype_typeobject(self):
        self.assertEqual(ndt.int8, nd.dtype(ctypes.c_int8))
        self.assertEqual(ndt.int16, nd.dtype(ctypes.c_int16))
        self.assertEqual(ndt.int32, nd.dtype(ctypes.c_int32))
        self.assertEqual(ndt.int64, nd.dtype(ctypes.c_int64))
        self.assertEqual(ndt.uint8, nd.dtype(ctypes.c_uint8))
        self.assertEqual(ndt.uint16, nd.dtype(ctypes.c_uint16))
        self.assertEqual(ndt.uint32, nd.dtype(ctypes.c_uint32))
        self.assertEqual(ndt.uint64, nd.dtype(ctypes.c_uint64))
        self.assertEqual(ndt.uint32, nd.dtype(ctypes.c_uint32))
        self.assertEqual(ndt.uint64, nd.dtype(ctypes.c_uint64))
        self.assertEqual(ndt.float32, nd.dtype(ctypes.c_float))
        self.assertEqual(ndt.float64, nd.dtype(ctypes.c_double))

    def test_dtype_from_annotated_ctype_typeobject(self):
        self.assertEqual(ndt.bool, nd.dtype(ndt.ctypes.c_dynd_bool))
        self.assertEqual(ndt.complex_float32, nd.dtype(ndt.ctypes.c_complex_float32))
        self.assertEqual(ndt.complex_float64, nd.dtype(ndt.ctypes.c_complex_float64))
        self.assertEqual(ndt.complex_float32, nd.dtype(ndt.ctypes.c_complex64))
        self.assertEqual(ndt.complex_float64, nd.dtype(ndt.ctypes.c_complex128))

    def test_dtype_from_ctype_cstruct(self):
        class POINT(ctypes.Structure):
            _fields_ = [('x', ctypes.c_int32), ('y', ctypes.c_int32)]
        self.assertEqual(ndt.make_fixedstruct_dtype(
                                [ndt.int32, ndt.int32],['x', 'y']),
                        nd.dtype(POINT))
        class DATA(ctypes.Structure):
            _fields_ = [
                        ('pos', POINT),
                        ('flags', ctypes.c_int8),
                        ('size', ctypes.c_float),
                        ('vel', POINT)
                       ]
        self.assertEqual(ndt.make_fixedstruct_dtype([POINT, ndt.int8, ndt.float32, POINT],
                                ['pos', 'flags', 'size', 'vel']),
                        nd.dtype(DATA))

    def test_dtype_from_ctypes_carray(self):
        self.assertEqual(ndt.make_fixed_dim_dtype(10, ndt.int32),
                nd.dtype(ctypes.c_int32 * 10))
        self.assertEqual(ndt.make_fixed_dim_dtype((10, 3), ndt.int32),
                nd.dtype((ctypes.c_int32 * 3) * 10))
        self.assertEqual(ndt.make_fixed_dim_dtype((10, 3, 4), ndt.int32),
                nd.dtype(((ctypes.c_int32 * 4) * 3) * 10))

        class POINT(ctypes.Structure):
            _fields_ = [('x', ctypes.c_int32), ('y', ctypes.c_int32)]
        self.assertEqual(ndt.make_fixed_dim_dtype(10, nd.dtype(POINT)),
                nd.dtype(POINT * 10))

if __name__ == '__main__':
    unittest.main()

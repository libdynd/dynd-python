import sys
import unittest
from blaze import nd
import ctypes

class TestCTypesDTypeInterop(unittest.TestCase):
    def test_dtype_from_ctype_typeobject(self):
        self.assertEqual(nd.int8, nd.dtype(ctypes.c_int8))
        self.assertEqual(nd.int16, nd.dtype(ctypes.c_int16))
        self.assertEqual(nd.int32, nd.dtype(ctypes.c_int32))
        self.assertEqual(nd.int64, nd.dtype(ctypes.c_int64))
        self.assertEqual(nd.uint8, nd.dtype(ctypes.c_uint8))
        self.assertEqual(nd.uint16, nd.dtype(ctypes.c_uint16))
        self.assertEqual(nd.uint32, nd.dtype(ctypes.c_uint32))
        self.assertEqual(nd.uint64, nd.dtype(ctypes.c_uint64))
        self.assertEqual(nd.uint32, nd.dtype(ctypes.c_uint32))
        self.assertEqual(nd.uint64, nd.dtype(ctypes.c_uint64))
        self.assertEqual(nd.float32, nd.dtype(ctypes.c_float))
        self.assertEqual(nd.float64, nd.dtype(ctypes.c_double))

    def test_dtype_from_annotated_ctype_typeobject(self):
        self.assertEqual(nd.bool, nd.dtype(nd.ctypes.c_dynd_bool))
        self.assertEqual(nd.complex_float32, nd.dtype(nd.ctypes.c_complex_float32))
        self.assertEqual(nd.complex_float64, nd.dtype(nd.ctypes.c_complex_float64))
        self.assertEqual(nd.complex_float32, nd.dtype(nd.ctypes.c_complex64))
        self.assertEqual(nd.complex_float64, nd.dtype(nd.ctypes.c_complex128))

if __name__ == '__main__':
    unittest.main()

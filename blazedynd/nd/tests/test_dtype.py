import sys
import unittest
from blazedynd import nd

class TestDType(unittest.TestCase):

    def test_bool_dtype_properties(self):
        self.assertEqual(type(nd.dt.bool), nd.dtype)
        self.assertEqual(str(nd.dt.bool), 'bool')
        self.assertEqual(nd.dt.bool.element_size, 1)
        self.assertEqual(nd.dt.bool.alignment, 1)

    def test_int_dtype_properties(self):
        self.assertEqual(type(nd.dt.int8), nd.dtype)
        self.assertEqual(str(nd.dt.int8), 'int8')
        self.assertEqual(nd.dt.int8.element_size, 1)
        self.assertEqual(nd.dt.int8.alignment, 1)

        self.assertEqual(type(nd.dt.int16), nd.dtype)
        self.assertEqual(str(nd.dt.int16), 'int16')
        self.assertEqual(nd.dt.int16.element_size, 2)
        self.assertEqual(nd.dt.int16.alignment, 2)

        self.assertEqual(type(nd.dt.int32), nd.dtype)
        self.assertEqual(str(nd.dt.int32), 'int32')
        self.assertEqual(nd.dt.int32.element_size, 4)
        self.assertEqual(nd.dt.int32.alignment, 4)

        self.assertEqual(type(nd.dt.int64), nd.dtype)
        self.assertEqual(str(nd.dt.int64), 'int64')
        self.assertEqual(nd.dt.int64.element_size, 8)
        self.assertEqual(nd.dt.int64.alignment, 8)

    def test_uint_dtype_properties(self):
        self.assertEqual(type(nd.dt.uint8), nd.dtype)
        self.assertEqual(str(nd.dt.uint8), 'uint8')
        self.assertEqual(nd.dt.uint8.element_size, 1)
        self.assertEqual(nd.dt.uint8.alignment, 1)

        self.assertEqual(type(nd.dt.uint16), nd.dtype)
        self.assertEqual(str(nd.dt.uint16), 'uint16')
        self.assertEqual(nd.dt.uint16.element_size, 2)
        self.assertEqual(nd.dt.uint16.alignment, 2)

        self.assertEqual(type(nd.dt.uint32), nd.dtype)
        self.assertEqual(str(nd.dt.uint32), 'uint32')
        self.assertEqual(nd.dt.uint32.element_size, 4)
        self.assertEqual(nd.dt.uint32.alignment, 4)

        self.assertEqual(type(nd.dt.uint64), nd.dtype)
        self.assertEqual(str(nd.dt.uint64), 'uint64')
        self.assertEqual(nd.dt.uint64.element_size, 8)
        self.assertEqual(nd.dt.uint64.alignment, 8)

    def test_float_dtype_properties(self):
        self.assertEqual(type(nd.dt.float32), nd.dtype)
        self.assertEqual(str(nd.dt.float32), 'float32')
        self.assertEqual(nd.dt.float32.element_size, 4)
        self.assertEqual(nd.dt.float32.alignment, 4)

        self.assertEqual(type(nd.dt.float64), nd.dtype)
        self.assertEqual(str(nd.dt.float64), 'float64')
        self.assertEqual(nd.dt.float64.element_size, 8)
        self.assertEqual(nd.dt.float64.alignment, 8)

    def test_complex_dtype_properties(self):
        self.assertEqual(type(nd.dt.cfloat32), nd.dtype)
        self.assertEqual(str(nd.dt.cfloat32), 'complex<float32>')
        self.assertEqual(nd.dt.cfloat32.element_size, 8)
        self.assertEqual(nd.dt.cfloat32.alignment, 4)

        self.assertEqual(type(nd.dt.cfloat64), nd.dtype)
        self.assertEqual(str(nd.dt.cfloat64), 'complex<float64>')
        self.assertEqual(nd.dt.cfloat64.element_size, 16)
        self.assertEqual(nd.dt.cfloat64.alignment, 8)

    def test_fixedstring_dtype_properties(self):
        d = nd.dt.make_fixedstring_dtype('ascii', 10)
        self.assertEqual(str(d), 'fixedstring<ascii,10>')
        self.assertEqual(d.element_size, 10)
        self.assertEqual(d.alignment, 1)
        self.assertEqual(d.encoding, 'ascii')

        d = nd.dt.make_fixedstring_dtype('ucs_2', 10)
        self.assertEqual(str(d), 'fixedstring<ucs_2,10>')
        self.assertEqual(d.element_size, 20)
        self.assertEqual(d.alignment, 2)
        self.assertEqual(d.encoding, 'ucs_2')

        d = nd.dt.make_fixedstring_dtype('utf_8', 10)
        self.assertEqual(str(d), 'fixedstring<utf_8,10>')
        self.assertEqual(d.element_size, 10)
        self.assertEqual(d.alignment, 1)
        self.assertEqual(d.encoding, 'utf_8')

        d = nd.dt.make_fixedstring_dtype('utf_16', 10)
        self.assertEqual(str(d), 'fixedstring<utf_16,10>')
        self.assertEqual(d.element_size, 20)
        self.assertEqual(d.alignment, 2)
        self.assertEqual(d.encoding, 'utf_16')

        d = nd.dt.make_fixedstring_dtype('utf_32', 10)
        self.assertEqual(str(d), 'fixedstring<utf_32,10>')
        self.assertEqual(d.element_size, 40)
        self.assertEqual(d.alignment, 4)
        self.assertEqual(d.encoding, 'utf_32')

    def test_scalar_dtypes(self):
        self.assertEqual(nd.dt.bool, nd.dtype(bool))
        self.assertEqual(nd.dt.int32, nd.dtype(int))
        self.assertEqual(nd.dt.float64, nd.dtype(float))
        self.assertEqual(nd.dt.cfloat64, nd.dtype(complex))

    def test_fixedbytes_dtype(self):
        d = nd.dt.make_fixedbytes_dtype(4, 4)
        self.assertEqual(str(d), 'fixedbytes<4,4>')
        self.assertEqual(d.element_size, 4)
        self.assertEqual(d.alignment, 4)

        d = nd.dt.make_fixedbytes_dtype(9, 1)
        self.assertEqual(str(d), 'fixedbytes<9,1>')
        self.assertEqual(d.element_size, 9)
        self.assertEqual(d.alignment, 1)

        # Alignment must not be greater than element_size
        self.assertRaises(RuntimeError, nd.dt.make_fixedbytes_dtype, 1, 2)
        # Alignment must be a power of 2
        self.assertRaises(RuntimeError, nd.dt.make_fixedbytes_dtype, 6, 3)
        # Alignment must divide into the element_size
        self.assertRaises(RuntimeError, nd.dt.make_fixedbytes_dtype, 6, 4)

if __name__ == '__main__':
    unittest.main()

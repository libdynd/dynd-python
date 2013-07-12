import sys
import unittest
from dynd import nd, ndt

class TestDType(unittest.TestCase):
    def test_bool_type_properties(self):
        self.assertEqual(type(ndt.bool), ndt.type)
        self.assertEqual(str(ndt.bool), 'bool')
        self.assertEqual(ndt.bool.data_size, 1)
        self.assertEqual(ndt.bool.data_alignment, 1)

    def test_int_type_properties(self):
        self.assertEqual(type(ndt.int8), ndt.type)
        self.assertEqual(str(ndt.int8), 'int8')
        self.assertEqual(ndt.int8.data_size, 1)
        self.assertEqual(ndt.int8.data_alignment, 1)

        self.assertEqual(type(ndt.int16), ndt.type)
        self.assertEqual(str(ndt.int16), 'int16')
        self.assertEqual(ndt.int16.data_size, 2)
        self.assertEqual(ndt.int16.data_alignment, 2)

        self.assertEqual(type(ndt.int32), ndt.type)
        self.assertEqual(str(ndt.int32), 'int32')
        self.assertEqual(ndt.int32.data_size, 4)
        self.assertEqual(ndt.int32.data_alignment, 4)

        self.assertEqual(type(ndt.int64), ndt.type)
        self.assertEqual(str(ndt.int64), 'int64')
        self.assertEqual(ndt.int64.data_size, 8)
        self.assertTrue(ndt.int64.data_alignment in [4,8])

    def test_uint_type_properties(self):
        self.assertEqual(type(ndt.uint8), ndt.type)
        self.assertEqual(str(ndt.uint8), 'uint8')
        self.assertEqual(ndt.uint8.data_size, 1)
        self.assertEqual(ndt.uint8.data_alignment, 1)

        self.assertEqual(type(ndt.uint16), ndt.type)
        self.assertEqual(str(ndt.uint16), 'uint16')
        self.assertEqual(ndt.uint16.data_size, 2)
        self.assertEqual(ndt.uint16.data_alignment, 2)

        self.assertEqual(type(ndt.uint32), ndt.type)
        self.assertEqual(str(ndt.uint32), 'uint32')
        self.assertEqual(ndt.uint32.data_size, 4)
        self.assertEqual(ndt.uint32.data_alignment, 4)

        self.assertEqual(type(ndt.uint64), ndt.type)
        self.assertEqual(str(ndt.uint64), 'uint64')
        self.assertEqual(ndt.uint64.data_size, 8)
        self.assertTrue(ndt.uint64.data_alignment in [4,8])

    def test_float_type_properties(self):
        self.assertEqual(type(ndt.float32), ndt.type)
        self.assertEqual(str(ndt.float32), 'float32')
        self.assertEqual(ndt.float32.data_size, 4)
        self.assertEqual(ndt.float32.data_alignment, 4)

        self.assertEqual(type(ndt.float64), ndt.type)
        self.assertEqual(str(ndt.float64), 'float64')
        self.assertEqual(ndt.float64.data_size, 8)
        self.assertTrue(ndt.float64.data_alignment in [4,8])

    def test_complex_type_properties(self):
        self.assertEqual(type(ndt.cfloat32), ndt.type)
        self.assertEqual(str(ndt.cfloat32), 'complex<float32>')
        self.assertEqual(ndt.cfloat32.data_size, 8)
        self.assertEqual(ndt.cfloat32.data_alignment, 4)

        self.assertEqual(type(ndt.cfloat64), ndt.type)
        self.assertEqual(str(ndt.cfloat64), 'complex<float64>')
        self.assertEqual(ndt.cfloat64.data_size, 16)
        self.assertTrue(ndt.cfloat64.data_alignment in [4,8])

    def test_complex_type_realimag(self):
        a = nd.array(1 + 3j)
        self.assertEqual(ndt.cfloat64, nd.type_of(a))
        self.assertEqual(1, nd.as_py(a.real))
        self.assertEqual(3, nd.as_py(a.imag))

        a = nd.array([1 + 2j, 3 + 4j, 5 + 6j])
        self.assertEqual(ndt.type('A, cfloat64'), nd.type_of(a))
        self.assertEqual([1, 3, 5], nd.as_py(a.real))
        self.assertEqual([2, 4, 6], nd.as_py(a.imag))

    def test_fixedstring_type_properties(self):
        d = ndt.make_fixedstring(10, 'ascii')
        self.assertEqual(str(d), "string<10,'ascii'>")
        self.assertEqual(d.data_size, 10)
        self.assertEqual(d.data_alignment, 1)
        self.assertEqual(d.encoding, 'ascii')

        d = ndt.make_fixedstring(10, 'ucs_2')
        self.assertEqual(str(d), "string<10,'ucs-2'>")
        self.assertEqual(d.data_size, 20)
        self.assertEqual(d.data_alignment, 2)
        self.assertEqual(d.encoding, 'ucs-2')

        d = ndt.make_fixedstring(10, 'utf-8')
        self.assertEqual(str(d), 'string<10>')
        self.assertEqual(d.data_size, 10)
        self.assertEqual(d.data_alignment, 1)
        self.assertEqual(d.encoding, 'utf-8')

        d = ndt.make_fixedstring(10, 'utf_16')
        self.assertEqual(str(d), "string<10,'utf-16'>")
        self.assertEqual(d.data_size, 20)
        self.assertEqual(d.data_alignment, 2)
        self.assertEqual(d.encoding, 'utf-16')

        d = ndt.make_fixedstring(10, 'utf_32')
        self.assertEqual(str(d), "string<10,'utf-32'>")
        self.assertEqual(d.data_size, 40)
        self.assertEqual(d.data_alignment, 4)
        self.assertEqual(d.encoding, 'utf-32')

    def test_scalar_types(self):
        self.assertEqual(ndt.bool, ndt.type(bool))
        self.assertEqual(ndt.int32, ndt.type(int))
        self.assertEqual(ndt.float64, ndt.type(float))
        self.assertEqual(ndt.cfloat64, ndt.type(complex))

    def test_fixedbytes_type(self):
        d = ndt.make_fixedbytes(4, 4)
        self.assertEqual(str(d), 'fixedbytes<4,4>')
        self.assertEqual(d.data_size, 4)
        self.assertEqual(d.data_alignment, 4)

        d = ndt.make_fixedbytes(9, 1)
        self.assertEqual(str(d), 'fixedbytes<9,1>')
        self.assertEqual(d.data_size, 9)
        self.assertEqual(d.data_alignment, 1)

        # Alignment must not be greater than data_size
        self.assertRaises(RuntimeError, ndt.make_fixedbytes, 1, 2)
        # Alignment must be a power of 2
        self.assertRaises(RuntimeError, ndt.make_fixedbytes, 6, 3)
        # Alignment must divide into the data_size
        self.assertRaises(RuntimeError, ndt.make_fixedbytes, 6, 4)

    def test_type_type(self):
        d = ndt.type('type')
        self.assertEqual(str(d), 'type')

        # Creating a dynd array out of a dtype
        # results in it having the dtype 'dtype'
        n = nd.array(d)
        self.assertEqual(nd.type_of(n), d)

        # Python float type converts to float64
        n = nd.array(float)
        self.assertEqual(nd.type_of(n), d)
        self.assertEqual(nd.as_py(n), ndt.float64)

    def test_cstruct_type(self):
        self.assertFalse(ndt.type('{x: int32}') == ndt.type('{y: int32}'))

    def test_categorical_type(self):
        a = nd.array(["2012-05-10T02:29:42Z"] * 100, "datetime('sec','UTC')")
        dt1 = ndt.factor_categorical(a.date)
        #print (dt1)
        self.assertEqual(nd.as_py(dt1.categories.ucast(ndt.string)),
                        ['2012-05-10'])

if __name__ == '__main__':
    unittest.main()

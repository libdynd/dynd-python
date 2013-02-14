import sys
import unittest
from dynd import nd, ndt
import numpy as np

class TestDTypeDataShape(unittest.TestCase):
    def test_scalars(self):
        """Tests making Blaze atom types from strings"""
        self.assertEqual(ndt.void, nd.dtype('void'))
        self.assertEqual(ndt.bool, nd.dtype('bool'))
        self.assertEqual(ndt.int8, nd.dtype('int8'))
        self.assertEqual(ndt.int16, nd.dtype('int16'))
        self.assertEqual(ndt.int32, nd.dtype('int32'))
        self.assertEqual(ndt.int64, nd.dtype('int64'))
        self.assertEqual(ndt.uint8, nd.dtype('uint8'))
        self.assertEqual(ndt.uint16, nd.dtype('uint16'))
        self.assertEqual(ndt.uint32, nd.dtype('uint32'))
        self.assertEqual(ndt.uint64, nd.dtype('uint64'))
        self.assertEqual(ndt.float32, nd.dtype('float32'))
        self.assertEqual(ndt.float64, nd.dtype('float64'))
        self.assertEqual(ndt.cfloat32, nd.dtype('cfloat32'))
        self.assertEqual(ndt.cfloat64, nd.dtype('cfloat64'))
        self.assertEqual(ndt.cfloat32, nd.dtype('complex64'))
        self.assertEqual(ndt.cfloat64, nd.dtype('complex128'))
        self.assertEqual(ndt.string, nd.dtype('string'))
        self.assertEqual(ndt.date, nd.dtype('date'))
        self.assertEqual(ndt.json, nd.dtype('json'))
        self.assertEqual(ndt.bytes, nd.dtype('bytes'))

    def test_fixed_array(self):
        """Tests of datashapes that produce the DyND fixed array dtype"""
        self.assertEqual(ndt.make_fixed_dim_dtype(3, ndt.int32),
                        nd.dtype('3, int32'))
        self.assertEqual(ndt.make_fixed_dim_dtype((5, 2), ndt.float64),
                        nd.dtype('5, 2, float64'))

if __name__ == '__main__':
    unittest.main()

import sys
import unittest
from dynd import nd, ndt
import numpy as np

class TestDTypeDataShape(unittest.TestCase):
    def test_scalars(self):
        # Tests making Blaze atom types from strings
        self.assertEqual(ndt.void, ndt.type('void'))
        self.assertEqual(ndt.bool, ndt.type('bool'))
        self.assertEqual(ndt.int8, ndt.type('int8'))
        self.assertEqual(ndt.int16, ndt.type('int16'))
        self.assertEqual(ndt.int32, ndt.type('int32'))
        self.assertEqual(ndt.int64, ndt.type('int64'))
        self.assertEqual(ndt.intptr, ndt.type('intptr'))
        self.assertEqual(ndt.uint8, ndt.type('uint8'))
        self.assertEqual(ndt.uint16, ndt.type('uint16'))
        self.assertEqual(ndt.uint32, ndt.type('uint32'))
        self.assertEqual(ndt.uint64, ndt.type('uint64'))
        self.assertEqual(ndt.uintptr, ndt.type('uintptr'))
        self.assertEqual(ndt.float32, ndt.type('float32'))
        self.assertEqual(ndt.float64, ndt.type('float64'))
        self.assertEqual(ndt.complex_float32, ndt.type('complex[float32]'))
        self.assertEqual(ndt.complex_float64, ndt.type('complex[float64]'))
        self.assertEqual(ndt.complex_float32, ndt.type('complex64'))
        self.assertEqual(ndt.complex_float64, ndt.type('complex128'))
        self.assertEqual(ndt.string, ndt.type('string'))
        self.assertEqual(ndt.date, ndt.type('date'))
        self.assertEqual(ndt.json, ndt.type('json'))
        self.assertEqual(ndt.bytes, ndt.type('bytes'))

    def test_aliases(self):
        self.assertEqual(ndt.int32, ndt.type('int'))
        self.assertEqual(ndt.float64, ndt.type('real'))
        self.assertEqual(ndt.complex_float64, ndt.type('complex'))

    def test_cfixed_array(self):
        # Tests of datashapes that produce the DyND fixed array type
        self.assertEqual(ndt.make_cfixed_dim(3, ndt.int32),
                        ndt.type('cfixed[3] * int32'))
        self.assertEqual(ndt.make_cfixed_dim((5, 2), ndt.float64),
                        ndt.type('cfixed[5] * cfixed[2] * float64'))

    def test_struct(self):
        # Tests of cstruct datashape
        dt = ndt.type('c{x: cfixed[3] * int32, y: string}')
        self.assertEqual(dt.type_id, 'cstruct')
        self.assertEqual(nd.as_py(dt.field_names), ['x', 'y'])
        # Tests of struct datashape
        dt = ndt.type('{x: 3 * int32, y: string}')
        self.assertEqual(dt.type_id, 'struct')
        self.assertEqual(nd.as_py(dt.field_names), ['x', 'y'])

    def test_var_dshape(self):
        # Getting the dshape can see into leading var dims
        a = nd.array([[[1], [2,3]]], type='var * var * var * int32')
        self.assertEqual(nd.dshape_of(a), '1 * 2 * var * int32')

if __name__ == '__main__':
    unittest.main()

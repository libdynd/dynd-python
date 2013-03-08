import sys
import unittest
from dynd import nd, ndt
# This feature depends on NumPy/SciPy for now
import numpy as np

class TestComputedFields(unittest.TestCase):

    def test_simple_expr(self):
        a = np.array([(1,), (2,), (3,), (4,), (5,)],
                dtype=[('xyz', np.int32)])
        b = nd.add_computed_fields(a,
                [('twice', np.int32, '2 * xyz'),
                 ('onemore', np.int16, 'xyz + 1')])
        self.assertEqual(b.dtype.element_dtype.type_id, 'unary_expr')
        self.assertEqual(b.dtype.element_dtype.value_dtype,
                nd.dtype('{xyz: int32; twice: int32; onemore: int16}'))
        self.assertEqual(nd.as_py(b.xyz), [1, 2, 3, 4, 5])
        self.assertEqual(nd.as_py(b.twice), [2, 4, 6, 8, 10])
        self.assertEqual(nd.as_py(b.onemore), [2, 3, 4, 5, 6])

    def test_rm_fields(self):
        a = np.array([(1, 2), (-1, 1), (2, 5)],
                dtype=[('x', np.float32), ('y', np.float32)])
        b = nd.add_computed_fields(a,
                fields=[('sum', np.float32, 'x + y'),
                        ('difference', np.float32, 'x - y'),
                        ('product', np.float32, 'x * y'),
                        ('complex', np.complex64, 'x + 1j*y')],
                rm_fields=['x', 'y'])
        self.assertEqual(b.dtype.element_dtype.value_dtype,
                nd.dtype('{sum: float32; difference: float32;' +
                    ' product: float32; complex: cfloat32}'))
        self.assertEqual(nd.as_py(b.sum), [3, 0, 7]),
        self.assertEqual(nd.as_py(b.difference), [-1, -2, -3])
        self.assertEqual(nd.as_py(b.product), [2, -1, 10])
        self.assertEqual(nd.as_py(b.complex), [1+2j, -1+1j, 2+5j])

if __name__ == '__main__':
    unittest.main()

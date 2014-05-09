import sys
import unittest
from dynd import nd, ndt
# This feature depends on NumPy/SciPy for now
import numpy as np

try:
    import scipy
except ImportError:
    scipy = None

if sys.version_info[:2] > (2, 6):
    class TestComputedFields(unittest.TestCase):
        @unittest.skipIf(scipy is None, "scipy is not installed")
        def test_simple_expr(self):
            a = np.array([(1,), (2,), (3,), (4,), (5,)],
                    dtype=[('xyz', np.int32)])
            b = nd.add_computed_fields(a,
                    [('twice', np.int32, '2 * xyz'),
                     ('onemore', np.int16, 'xyz + 1')])
            self.assertEqual(nd.type_of(b).element_type.type_id, 'unary_expr')
            self.assertEqual(nd.type_of(b).element_type.value_type,
                    ndt.type('c{xyz: int32, twice: int32, onemore: int16}'))
            self.assertEqual(nd.as_py(b.xyz), [1, 2, 3, 4, 5])
            self.assertEqual(nd.as_py(b.twice), [2, 4, 6, 8, 10])
            self.assertEqual(nd.as_py(b.onemore), [2, 3, 4, 5, 6])

        @unittest.skipIf(scipy is None, "scipy is not installed")
        def test_rm_fields(self):
            a = np.array([(1, 2), (-1, 1), (2, 5)],
                    dtype=[('x', np.float32), ('y', np.float32)])
            b = nd.add_computed_fields(a,
                    fields=[('sum', np.float32, 'x + y'),
                            ('difference', np.float32, 'x - y'),
                            ('product', np.float32, 'x * y'),
                            ('complex', np.complex64, 'x + 1j*y')],
                    rm_fields=['x', 'y'])
            self.assertEqual(nd.type_of(b).element_type.value_type,
                    ndt.type('c{sum: float32, difference: float32,' +
                        ' product: float32, complex: complex[float32]}'))
            self.assertEqual(nd.as_py(b.sum), [3, 0, 7]),
            self.assertEqual(nd.as_py(b.difference), [-1, -2, -3])
            self.assertEqual(nd.as_py(b.product), [2, -1, 10])
            self.assertEqual(nd.as_py(b.complex), [1+2j, -1+1j, 2+5j])

        @unittest.skipIf(scipy is None, "scipy is not installed")
        def test_aggregate(self):
            a = nd.array([
                ('A', 1, 2),
                ('A', 3, 4),
                ('B', 1.5, 2.5),
                ('A', 0.5, 9),
                ('C', 1, 5),
                ('B', 2, 2)],
                dtype='c{cat: string, x: float32, y: float32}')
            gb = nd.groupby(a, nd.fields(a, 'cat')).eval()
            b = nd.make_computed_fields(gb, 1,
                    fields=[('sum_x', ndt.float32, 'sum(x)'),
                            ('mean_y', ndt.float32, 'mean(y)'),
                            ('max_x', ndt.float32, 'max(x)'),
                            ('max_y', ndt.float32, 'max(y)')])
            self.assertEqual(nd.as_py(b.sum_x), [4.5, 3.5, 1])
            self.assertEqual(nd.as_py(b.mean_y), [5, 2.25, 5])
            self.assertEqual(nd.as_py(b.max_x), [3, 2, 1])
            self.assertEqual(nd.as_py(b.max_y), [9, 2.5, 5])

if __name__ == '__main__':
    unittest.main(verbosity=2)

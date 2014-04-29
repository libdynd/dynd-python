import unittest
from dynd import nd, ndt

class TestDimHelpers(unittest.TestCase):
    def test_create(self):
        self.assertEqual(ndt.var * ndt.int32, ndt.type('var * int32'))
        self.assertEqual(ndt.strided * ndt.int32, ndt.type('strided * int32'))
        self.assertEqual(ndt.fixed[7] * ndt.int32, ndt.type('7 * int32'))
        self.assertEqual(ndt.cfixed[6] * ndt.int32,
                         ndt.type('cfixed[6] * int32'))
        self.assertEqual(ndt.strided * ndt.strided * ndt.float64,
                         ndt.type('strided * strided * float64'))
        self.assertEqual(ndt.strided ** 3 * ndt.float32,
                         ndt.type('strided * strided * strided * float32'))
        self.assertEqual((ndt.strided * ndt.fixed[2]) ** 2 * ndt.int16,
                         ndt.type('strided * 2 * strided * 2 * int16'))

    def test_create_fromtype(self):
        self.assertEqual(ndt.strided * int, ndt.type('strided * int'))
        self.assertEqual(ndt.strided * float, ndt.type('strided * real'))
        self.assertEqual(ndt.strided * complex, ndt.type('strided * complex'))
        self.assertEqual(ndt.strided * str, ndt.type('strided * string'))

    def test_create_struct(self):
        self.assertEqual(ndt.strided * ndt.var * '{x : int32, y : float32}',
                         ndt.type('strided * var * {x : int32, y : float32}'))

    def test_repr(self):
        self.assertEqual(repr(ndt.var), 'ndt.var')
        self.assertEqual(repr(ndt.strided), 'ndt.strided')
        self.assertEqual(repr(ndt.fixed), 'ndt.fixed')
        self.assertEqual(repr(ndt.fixed[5]), 'ndt.fixed[5]')
        self.assertEqual(repr(ndt.cfixed[12]), 'ndt.cfixed[12]')
        # Dimension fragment
        self.assertEqual(repr(ndt.var * ndt.strided *
                              ndt.fixed[2] * ndt.cfixed[3]),
                         'ndt.var * ndt.strided * ndt.fixed[2] * ndt.cfixed[3]')

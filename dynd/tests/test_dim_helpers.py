import unittest
from dynd import nd, ndt

class TestDimHelpers(unittest.TestCase):
    def test_create(self):
        self.assertEqual(ndt.var * ndt.int32, ndt.type('var * int32'))
        self.assertEqual(ndt.fixed * ndt.int32, ndt.type('fixed * int32'))
        self.assertEqual(ndt.fixed[7] * ndt.int32, ndt.type('7 * int32'))
        self.assertEqual(ndt.cfixed[6] * ndt.int32,
                         ndt.type('cfixed[6] * int32'))
        self.assertEqual(ndt.fixed * ndt.fixed * ndt.float64,
                         ndt.type('fixed * fixed * float64'))
        self.assertEqual(ndt.fixed ** 3 * ndt.float32,
                         ndt.type('fixed * fixed * fixed * float32'))
        self.assertEqual((ndt.fixed * ndt.fixed[2]) ** 2 * ndt.int16,
                         ndt.type('fixed * 2 * fixed * 2 * int16'))

    def test_create_fromtype(self):
        self.assertEqual(ndt.fixed * int, ndt.type('fixed * int'))
        self.assertEqual(ndt.fixed * float, ndt.type('fixed * real'))
        self.assertEqual(ndt.fixed * complex, ndt.type('fixed * complex'))
        self.assertEqual(ndt.fixed * str, ndt.type('fixed * string'))

    def test_create_struct(self):
        self.assertEqual(ndt.fixed * ndt.var * '{x : int32, y : float32}',
                         ndt.type('fixed * var * {x : int32, y : float32}'))

    def test_repr(self):
        self.assertEqual(repr(ndt.var), 'ndt.var')
        self.assertEqual(repr(ndt.fixed), 'ndt.fixed')
        self.assertEqual(repr(ndt.fixed), 'ndt.fixed')
        self.assertEqual(repr(ndt.fixed[5]), 'ndt.fixed[5]')
        self.assertEqual(repr(ndt.cfixed[12]), 'ndt.cfixed[12]')
        # Dimension fragment
        self.assertEqual(repr(ndt.var * ndt.fixed *
                              ndt.fixed[2] * ndt.cfixed[3]),
                         'ndt.var * ndt.fixed * ndt.fixed[2] * ndt.cfixed[3]')

import unittest
from dynd import nd, ndt

class TestDimHelpers(unittest.TestCase):
    def test_create(self):
        self.assertEqual(ndt.var * ndt.int32, ndt.type('var * int32'))
        self.assertEqual(ndt.fixed * ndt.int32, ndt.type('Fixed * int32'))
        self.assertEqual(ndt.fixed[7] * ndt.int32, ndt.type('7 * int32'))
        self.assertEqual(ndt.fixed[6] * ndt.int32,
                         ndt.type('fixed[6] * int32'))
        self.assertEqual(ndt.fixed * ndt.fixed * ndt.float64,
                         ndt.type('Fixed * Fixed * float64'))
        self.assertEqual(ndt.fixed ** 3 * ndt.float32,
                         ndt.type('Fixed * Fixed * Fixed * float32'))
        self.assertEqual((ndt.fixed * ndt.fixed[2]) ** 2 * ndt.int16,
                         ndt.type('Fixed * 2 * Fixed * 2 * int16'))

    def test_create_fromtype(self):
        self.assertEqual(ndt.fixed * int, ndt.type('Fixed * int'))
        self.assertEqual(ndt.fixed * float, ndt.type('Fixed * real'))
        self.assertEqual(ndt.fixed * complex, ndt.type('Fixed * complex'))
        self.assertEqual(ndt.fixed * str, ndt.type('Fixed * string'))

    def test_create_struct(self):
        self.assertEqual(ndt.fixed * ndt.var * '{x : int32, y : float32}',
                         ndt.type('Fixed * var * {x : int32, y : float32}'))

    def test_repr(self):
        self.assertEqual(repr(ndt.var), 'ndt.var')
        self.assertEqual(repr(ndt.fixed), 'ndt.fixed')
        self.assertEqual(repr(ndt.fixed), 'ndt.fixed')
        self.assertEqual(repr(ndt.fixed[5]), 'ndt.fixed[5]')
        # Dimension fragment
        self.assertEqual(repr(ndt.var * ndt.fixed *
                              ndt.fixed[2] * ndt.fixed[3]),
                         'ndt.var * ndt.fixed * ndt.fixed[2] * ndt.fixed[3]')
